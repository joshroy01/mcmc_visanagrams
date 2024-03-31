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
