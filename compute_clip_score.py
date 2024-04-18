import torch
import clip
from PIL import Image

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


root_dir = 'C:\\Users\\ferdawsi\\Documents\\EECS_542\\va_reproduction\\'
root_prompt = "a painting of "

text_prompts = [["a duck", "a fish"], ["a penguin", "a giraffe"], ["a flower", "a dog"], ["an old woman","a young lady"], ["a red panda", "kitchenware"]] 
index = 0

image_paths = ["duck_2.png", "fish_2.png", "penguin_2.png", "giraffe_2.png", "flower_2.png", "dog_2.png", "old_2.png", "young_2.png", "panda_2.png", "kitchen_2.png"]
for k, dir in enumerate(image_paths):
    image_path = root_dir + dir
    image = Image.open(image_path)

    # Define texts and rotations

    texts = text_prompts[index]
    texts = [root_prompt + item for item in texts]

    rotations = ["identical", "flip"]

    # Create a 2x2 tensor to store scores
    scores_tensor = torch.zeros((2, 2))

    # Iterate over rotations
    for i, rotation in enumerate(rotations):
        # Iterate over texts
        for j, text in enumerate(texts):
            # Calculate CLIP score
            score = get_clip_score(image, text, rotation)
            scores_tensor[i, j] = score

    # Save the scores tensor as a .pth file
    output_filename = root_dir + "stage_2\\" + str(k) + "_scores.pth"
    torch.save(scores_tensor, output_filename)
    if (k % 2 != 0):
        index = index + 1 



# root_dir = 'C:\\Users\\ferdawsi\\Documents\\EECS_542\\va_reproduction\\'
# root_prompt = "a painting of "

# image_paths = ["duck_2.png", "fish_2.png", "penguin_2.png", "giraffe_2.png", "flower_2.png", "dog_2.png", "old_2.png", "young_2.png", "panda_2.png", "kitchen_2.png"]
# text_prompts = ["a duck", "a fish", "a penguin", "a giraffe", "a flower", "a dog", "an old woman","a young lady", "a red panda", "kitchenware"] 

# # Concatenate root_dir to image_paths
# image_paths = [root_dir + item for item in image_paths]

# # Concatenate root_prompt to text_prompts
# text_prompts = [root_prompt + item for item in text_prompts]

# # Get the CLIP scores tensor
# clip_scores_tensor = get_clip_scores(image_paths, text_prompts)

# # Save the scores tensor to .pth file
# save_scores(clip_scores_tensor, "stage1_scores.pth")
