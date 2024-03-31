import torch
from torch.nn.functional import interpolate


def make_canvas(latents, canvas_size, sizes, in_channels=3, base_size=64):
    # Make a canvas from different latents
    canvas_count = torch.zeros(canvas_size, canvas_size).to(latents.device)
    canvas_latent = torch.zeros(1, in_channels, canvas_size, canvas_size,
                                dtype=latents.dtype).to(latents.device)

    for size, latent in zip(sizes, latents):
        latent = latent[None]
        sf, x_start, y_start = size
        size = min(canvas_size - x_start, base_size) * sf
        latent_expand = interpolate(latent, (size, size), mode='nearest')

        weight = 1 / (sf**2)
        coords = torch.linspace(-1, 1, size).to(latents.device)
        XX, YY = torch.meshgrid(coords, coords)
        dist_from_edge = torch.sqrt(torch.min((1 - torch.abs(XX))**2, (1 - torch.abs(YY))**2))
        dist_from_edge = torch.minimum(
            torch.tensor(6 / 32), dist_from_edge
        )  # only fall off along the outer edge, to avoid sharp lines at edge of each contributing image
        dist_from_edge = dist_from_edge / torch.max(dist_from_edge)
        weight = weight * dist_from_edge + 1e-6

        canvas_latent[:, :, y_start:y_start + size, x_start:x_start +
                      size] = weight * latent_expand + canvas_latent[:, :, y_start:y_start + size,
                                                                     x_start:x_start + size]
        canvas_count[y_start:y_start + size, x_start:x_start +
                     size] = canvas_count[y_start:y_start + size, x_start:x_start + size] + weight

    canvas_latent = canvas_latent / (canvas_count[None, None] + 1e-10)
    canvas_latent = torch.nan_to_num(canvas_latent)
    return canvas_latent
