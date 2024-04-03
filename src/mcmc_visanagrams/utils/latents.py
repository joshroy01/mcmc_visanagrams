import torch
from torch.nn.functional import interpolate


# Code for take individual latents grids and making them into a 2D canvas from a set of conditioned text instructions
def extract_latents(latent_canvas, sizes):
    # Extract different latent chunks from a canvas
    latents = []

    for size in sizes:
        sf, x_start, y_start = size
        width = sf * 64
        latent = latent_canvas[:, :, y_start:y_start + width, x_start:x_start + width]

        if latent.shape[-1] == 64:
            latent = latent
        else:
            latent = interpolate(latent, (64, 64), mode='nearest')

        latents.append(latent)

    latents = torch.cat(latents, dim=0).type(latent_canvas.dtype)
    return latents

def extract_latents_stage_2(latent_canvas, sizes, target_size=128):
    """Extracts latents for stage 2

    NOTE: This should eventually be combined with the extract_latents function
    """
    latents = []
    for size in sizes:
        sf, x_start, y_start = size
        width = sf * target_size
        latent = latent_canvas[:, :, y_start:y_start + width, x_start:x_start + width]

        if latent.shape[-1] == target_size:
            latent = latent
        else:
            latent = interpolate(latent, (target_size, target_size), mode='nearest')

        latents.append(latent)

    latents = torch.cat(latents, dim=0).type(latent_canvas.dtype)
    return latents