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
from mcmc_visanagrams.utils.display import image_from_latents
from mcmc_visanagrams.utils.latents import extract_latents
from mcmc_visanagrams.utils.report import make_report
from mcmc_visanagrams.runners.config import Config

# Code for Samplers
from mcmc_visanagrams.samplers.annealed_ula_sampler import AnnealedULASampler
from mcmc_visanagrams.samplers.annealed_uha_sampler import AnnealedUHASampler


def save_all_views_of_latent(latent_canvas: torch.Tensor, sizes, views, output_path: Path,
                             target_size: int):
    sub_latents = extract_latents(latent_canvas, sizes=sizes, views=views, target_size=target_size)
    saved_img_paths = []
    for i, sub_lat in enumerate(sub_latents):
        img = image_from_latents(sub_lat.unsqueeze(0), clip_dynamic_range=True)
        fig, ax = plt.subplots()
        ax.imshow(img)

        view_name = type(views[i]).__name__
        ax.set_title(view_name)

        img_path = output_path / f"output_{i}_{view_name}.png"
        fig.savefig(img_path)

        img_np_path = output_path / f"output_{i}_{view_name}.npy"
        np.save(img_np_path, img)

        saved_img_paths.append(img_path)
    return saved_img_paths


def run_full_pipeline(config: Config):

    from mcmc_visanagrams.utils.display import visualize_context

    # guidance_mag = 20.0
    # guidance_mag = 10.0

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
            base_img_size=config.stage_1_args["base_img_size"],
            generator=generator,
            num_inference_steps=config.stage_1_args["num_inference_steps"],
            mcmc_iteration_cutoff=config.stage_1_args["mcmc_iteration_cutoff"],
            using_va_method=config.stage_1_args["using_va_method"],
            using_mcmc_sampling=config.stage_1_args["using_mcmc_sampling"])

    torch.save(latent_canvas_stage_1,
               config.stage_1_output_path / 'latent_canvas_stage_1_output.pt')

    sizes = config.context_list.collapse_sizes()
    views = config.context_list.collapse_views()
    saved_img_paths = save_all_views_of_latent(latent_canvas_stage_1,
                                               sizes,
                                               views,
                                               config.stage_1_output_path,
                                               target_size=config.stage_1_args["base_img_size"])
    make_report(config, saved_img_paths, stage_num=1)

    if config.model_spec["stage_2"] is None:
        print("Skipping stage 2 as model spec is not provided.")
        return

    stage_2 = IFSuperResolutionPipeline.from_pretrained(config.model_spec["stage_2"],
                                                        text_encoder=None,
                                                        variant="fp16",
                                                        torch_dtype=torch.float16)
    stage_2.enable_xformers_memory_efficient_attention()
    stage_2.enable_model_cpu_offload()
    stage_2.text_encoder = stage_1.text_encoder

    with torch.no_grad():
        latent_canvas_stage_2 = stage_2(
            image=latent_canvas_stage_1,
            context=config.context_list,
            sampler=la_sampler,
            height=config.stage_2_args["height"],
            width=config.stage_2_args["width"],
            #  prompt_embeds=prompt_embeds,
            #  negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
            num_inference_steps=config.stage_2_args["num_inference_steps"],
            mcmc_iteration_cutoff=config.stage_2_args["mcmc_iteration_cutoff"],
            base_img_size=config.stage_2_args["base_img_size"],
            noise_level=config.stage_2_args["noise_level"],
            using_va_method=config.stage_2_args["using_va_method"],
            using_mcmc_sampling=config.stage_2_args["using_mcmc_sampling"])

    torch.save(latent_canvas_stage_2,
               config.stage_2_output_path / 'latent_canvas_stage_2_output.pt')

    sizes = config.context_list.collapse_sizes(is_stage_2=True)
    saved_img_paths = save_all_views_of_latent(latent_canvas_stage_2,
                                               sizes,
                                               views,
                                               config.stage_2_output_path,
                                               target_size=config.stage_2_args["base_img_size"])
    make_report(config, saved_img_paths, stage_num=2)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to the configuration file")

    args = parser.parse_args()

    config = Config(args.config_path)
    run_full_pipeline(config)
