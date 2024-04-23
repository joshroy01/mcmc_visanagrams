from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
REPO_DIR = FILE_DIR.parent

import torch
import clip
from PIL import Image
import yaml
import numpy as np


def get_clip_score(image, text, rotation):
    # Load the pre-trained CLIP model
    model, preprocess = clip.load('ViT-B/32')

    # Define rotation angles
    rotation_map = {"identical": 0, "flip": 180}

    # Get the rotation angle
    rotation_angle = rotation_map.get(rotation.lower())
    if rotation_angle is None:
        raise ValueError("Rotation must be 'identical' or 'flip'")

    # Rotate the image if necessary
    if rotation_angle != 0:
        image = image.rotate(rotation_angle)

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()

    return clip_score


def main():
    root_path = REPO_DIR / "output" / "multirun_test" / "sumaiya_output"
    prompt_shortnames = [
        "dog_flower", "fish_duck", "penguin_giraffe", "kitchen_panda", "young_old_woman"
    ]

    methods = ["mcmc", "va_reproduction", "va_unconditional", "mcmc_unconditional"]
    results = dict()
    for prompt_shortname in prompt_shortnames:
        prompt_root = root_path / prompt_shortname
        results[prompt_shortname] = dict()
        for method in methods:

            method_root = prompt_root / method
            results[prompt_shortname][method] = torch.zeros((2, 2))

            for seed_dir in method_root.glob("seed_*"):
                seed = seed_dir.name.split("_")[1]

                # Read the config file to determine the actual prompts.
                config_path = seed_dir / "config.yaml"
                with config_path.open("r") as f:
                    config = yaml.safe_load(f)
                # Create a map between index and associated prompt.
                prompt_view_map = [[c["prompt"], c["view"]] for c in config["context_list"]]
                for i in range(len(prompt_view_map)):
                    view_str = prompt_view_map[i][1]
                    # results[prompt_shortname][method][view_str] = 0
                    if view_str == "identity":
                        prompt_view_map[i][1] = "identical"
                    elif prompt_view_map[i][1] == "rotate_180":
                        prompt_view_map[i][1] = "flip"

                # Create a 2x2 tensor to store scores
                scores_tensor = torch.zeros((2, 2))

                stage_2_output_path = seed_dir / "stage_2"

                views = [p[1] for p in prompt_view_map]
                texts = [p[0] for p in prompt_view_map]

                # Lazily construct the image matrix
                img_mat = [[None, None], [None, None]]
                path_00 = stage_2_output_path / f"output_0_IdentityView.npy"
                img_mat[0][0] = Image.fromarray(np.load(path_00))
                img_mat[1][0] = Image.fromarray(np.load(path_00))
                path_11 = stage_2_output_path / f"output_1_Rotate180View.npy"
                img_mat[0][1] = Image.fromarray(np.load(path_11))
                img_mat[1][1] = Image.fromarray(np.load(path_11))

                for i, rotation in enumerate(views):
                    for j, text in enumerate(texts):
                        score = get_clip_score(img_mat[i][j], text, rotation)
                        print(
                            f"Seed: {seed}, Method: {method}, Prompt: {text}, Score: {score}, view: {rotation}"
                        )

                        scores_tensor[i, j] = score
                results[prompt_shortname][method] += scores_tensor / 3.  # average over num seeds.

            # Save the scores tensor as a .pth file
            output_filename = method_root / "clip_scores.pth"
            torch.save(results[prompt_shortname][method], output_filename)


if __name__ == "__main__":
    main()
