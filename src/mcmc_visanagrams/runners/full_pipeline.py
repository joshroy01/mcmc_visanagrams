from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import autocast
from diffusers import DDIMScheduler, DiffusionPipeline
import yaml

from mcmc_visanagrams.pipelines.if_pipeline import IFPipeline
from mcmc_visanagrams.pipelines.if_super_resolution_pipeline import IFSuperResolutionPipeline
from mcmc_visanagrams.utils.output import save_model_spec, save_context
from mcmc_visanagrams.context import ContextList, Context
from mcmc_visanagrams.utils.display import image_from_latents
from mcmc_visanagrams.utils.latents import extract_latents
from mcmc_visanagrams.utils.report import make_report

# Code for Samplers
from mcmc_visanagrams.samplers.annealed_ula_sampler import AnnealedULASampler
from mcmc_visanagrams.samplers.annealed_uha_sampler import AnnealedUHASampler

# from huggingface_hub import notebook_login

# notebook_login()

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT_PATH = REPO_ROOT / "output"
OUTPUT_ROOT_PATH.mkdir(exist_ok=True)


class Config:

    def __init__(self, config_path: Path):
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        self._config_dict = config

        self.model_spec = config["model_spec"]
        self.trial_name: str = config["trial_name"]
        self.trial_output_path: Path = OUTPUT_ROOT_PATH / self.trial_name
        self.stage_1_output_path: Path = self.trial_output_path / "stage_1"
        self.stage_2_output_path: Path = self.trial_output_path / "stage_2"
        self.context_list = self._context_list_from_config(config)
        self.sampler = config["sampler"]
        self.seed = config.get("seed", None)
        self.stage_1_args = config["stage_1_args"]

        # Make the trial output path and save the config file in it.
        self.trial_output_path.mkdir(exist_ok=True)
        self.stage_1_output_path.mkdir(exist_ok=True)
        self.stage_2_output_path.mkdir(exist_ok=True)
        with (self.trial_output_path / "config.yaml").open("w") as f:
            yaml.safe_dump(config, f)

    def _context_list_from_config(self, config):
        context_list = ContextList()
        for context in config["context_list"]:
            size = tuple(context.pop("size"))
            context_list.append(Context(size=size, **context))
        return context_list

    # def initialize_sampler(self):
    #     if self.sampler["name"] == "AnnealedULASampler":
    #         return AnnealedULASampler(self.sampler["num_steps"], self.sampler["num_samplers_per_steps"])


def main(config: Config):

    from mcmc_visanagrams.utils.display import visualize_context

    # guidance_mag = 20.0
    guidance_mag = 10.0

    # Specify the locations of textual descriptions to compose
    # The keys have the form (scale, x, y) where scale is the size of the canvas and x, y is the starting locations

    # TODO: Visualize context here.

    # color_lookup = {}
    # np.random.seed(1)
    # for k, v in context.items():
    #     color_lookup[v['string']] = (np.random.uniform(size=(3, )), k[0]**2)

    # plt.figure(figsize=(5, 5))
    # img = visualize_context(128, 64, context, color_lookup)

    # plt.imshow(img)

    # for k, v in context.items():
    #     scale, xstart, ystart = k
    #     caption = v['string']
    #     color = color_lookup[caption][0]
    #     plt.plot([], [], color=color, label=caption)

    # plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    # plt.savefig('composite_captions.pdf', bbox_inches='tight')
    # plt.savefig('composite_captions.png', bbox_inches='tight', facecolor=plt.gca().get_facecolor())
    # # %download_file composite_captions.pdf
    # # %download_file composite_captions.png

    has_cuda = torch.cuda.is_available()
    device = torch.device('cpu' if not has_cuda else 'cuda')
    print(device)

    # initialize model
    stage_1 = IFPipeline.from_pretrained(config.model_spec["stage_1"],
                                         variant="fp16",
                                         torch_dtype=torch.float16,
                                         use_auth_token=True)
    stage_1.enable_xformers_memory_efficient_attention()
    stage_1.enable_model_cpu_offload()
    stage_1.safety_checker = None

    # Initialize sampler from config
    if config.sampler["name"] == "AnnealedULASampler":
        la_step_sizes = stage_1.scheduler.betas * 2

        alphas = 1 - stage_1.scheduler.betas
        alphas_cumprod = np.cumprod(alphas)
        scalar = np.sqrt(1 / (1 - alphas_cumprod))

        la_sampler = AnnealedULASampler(config.sampler["kwargs"]["num_steps"],
                                        config.sampler["kwargs"]["num_samples_per_step"],
                                        la_step_sizes, None, None, None)
    else:
        raise NotImplementedError("Only AnnealedULASampler is supported at the moment.")

    generator = torch.Generator('cuda')
    if config.seed:
        generator.manual_seed(config.seed)

    with torch.no_grad():
        latent_canvas_stage_1 = stage_1(
            config.context_list,
            la_sampler,
            height=config.stage_1_args["height"],
            width=config.stage_1_args["width"],
            generator=generator,
            num_inference_steps=config.stage_1_args["num_inference_steps"])

    torch.save(latent_canvas_stage_1,
               config.stage_1_output_path / 'latent_canvas_stage_1_output.pt')

    sizes = config.context_list.collapse_sizes()
    views = config.context_list.collapse_views()
    sub_latents = extract_latents(latent_canvas_stage_1, sizes=sizes, views=views)
    saved_img_paths = []
    for i, sub_lat in enumerate(sub_latents):
        img = image_from_latents(sub_lat.unsqueeze(0), clip_dynamic_range=True)
        fig, ax = plt.subplots()
        ax.imshow(img)

        view_name = type(views[i]).__name__
        ax.set_title(view_name)

        img_path = config.stage_1_output_path / f"sub_latent_{i}.png"
        fig.savefig(img_path)
        saved_img_paths.append(img_path)

    make_report(config, saved_img_paths, stage_num=1)

    # stage_2 = IFSuperResolutionPipeline.from_pretrained(config.model_spec["stage_2"],
    #                                                     text_encoder=None,
    #                                                     variant="fp16",
    #                                                     torch_dtype=torch.float16)
    # stage_2.enable_xformers_memory_efficient_attention()
    # stage_2.enable_model_cpu_offload()

    # # num_steps_stage_2 = 50
    # num_steps_stage_2 = steps
    # stage_2.text_encoder = stage_1.text_encoder

    # with torch.no_grad():
    #     images = stage_2(
    #         image=latent_canvas_stage_1,
    #         context=context,
    #         sampler=la_sampler,
    #         height=256,
    #         width=256,
    #         #  prompt_embeds=prompt_embeds,
    #         #  negative_prompt_embeds=negative_embeds,
    #         generator=generator,
    #         output_type="pt",
    #         num_inference_steps=num_steps_stage_2)

    # # save upsampled image
    # if not isinstance(images, np.ndarray):
    #     images = images[0].cpu().numpy().transpose(1, 2, 0)
    #     images = ((images + 1) / 2 * 255)

    # if CLIP_DYNAMIC_RANGE:
    #     images[images < 0.0] = 0.0
    #     images[images > 255] = 255

    # print(images.min())
    # print(images.max())

    # images = images.astype(np.uint8)
    # plt.imshow(images)

    # # np.save("oil_painting_swiss_alps_no_composed_diffusion_2.npy", images)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to the configuration file")

    args = parser.parse_args()

    config = Config(args.config_path)
    main(config)