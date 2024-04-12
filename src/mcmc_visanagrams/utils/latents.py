from typing import TYPE_CHECKING, List, Tuple, Optional

import torch
from torch.nn.functional import interpolate

if TYPE_CHECKING:
    from mcmc_visanagrams.views.view_base import BaseView


# Code for take individual latents grids and making them into a 2D canvas from a set of conditioned text instructions
def extract_latents(latent_canvas: torch.Tensor,
                    sizes: List[Tuple[int, int, int]],
                    views: Optional[List['BaseView']] = None,
                    target_size: int = 64) -> torch.Tensor:
    # Extract different latent chunks from a canvas
    latents = []

    for i, size in enumerate(sizes):
        sf, x_start, y_start = size
        width = sf * target_size
        latent = latent_canvas[:, :, y_start:y_start + width, x_start:x_start + width]

        if views:
            latent_viewed = views[i].view(latent)
        else:
            latent_viewed = latent

        if latent_viewed.shape[-1] == target_size:
            latent_viewed = latent_viewed
        else:
            latent_viewed = interpolate(latent_viewed, (target_size, target_size), mode='nearest')

        latents.append(latent_viewed)

    latents = torch.cat(latents, dim=0).type(latent_canvas.dtype)
    return latents


def extract_latents_stage_2(latent_canvas: torch.Tensor,
                            sizes,
                            views: Optional[List['BaseView']] = None,
                            target_size: int = 128) -> torch.Tensor:
    """Extracts latents from the composed canvas for stage 2

    Returns a tensor that is [num_contexts, num_channels, 128, 128]

    NOTE: This should eventually be combined with the extract_latents function
    """
    if latent_canvas.shape[-1] != 256:
        raise ValueError("Latent canvas should have a width of 256")
    if latent_canvas.shape[-2] != 256:
        raise ValueError("Latent canvas should have a height of 256")
    latents = extract_latents(latent_canvas, sizes, views, target_size=target_size)
    # latents = []
    # for size in sizes:
    #     sf, x_start, y_start = size
    #     width = sf * target_size
    #     latent = latent_canvas[:, :, y_start:y_start + width, x_start:x_start + width]

    #     if latent.shape[-1] == target_size:
    #         latent = latent
    #     else:
    #         # NOTE: According to PyTorch documentation, mode='nearest' is buggy. It is preferred to
    #         # use 'nearest-exact' instead.
    #         latent = interpolate(latent, (target_size, target_size), mode='nearest-exact')

    #     latents.append(latent)

    # latents = torch.cat(latents, dim=0).type(latent_canvas.dtype)
    return latents


def test_canvas_and_extraction(latents, context):
    raise NotImplementedError(
        "This function was included more as an example of how one would test the canvas and "
        "extraction functions. It is not meant to be run as is.")
    latents_upscaled = torch.nn.functional.interpolate(latents,
                                                       size=(256, 256),
                                                       mode="bilinear",
                                                       align_corners=True)
    sizes = [[k[0], *[v * 2 for v in k[1:]]] for k in context.keys()]

    latent_canvas_test = latents_upscaled

    for i in range(20):
        lats_test = extract_latents_stage_2(latent_canvas_test, sizes)
        print("lats_test shape:", lats_test.shape)

        # lats_test_np = lats_test.detach().cpu().numpy()

        # for i in range(lats_test.shape[0]):
        #     image = image_from_latents(lats_test[i].unsqueeze(0))
        #     plt.imshow(image)
        #     plt.show()

        # Making the canvas from the extracted latents
        latent_canvas_test = make_canvas_stage_2(lats_test, 256, sizes, in_channels=3)
    canvas = image_from_latents(latent_canvas_test)
    plt.figure()
    plt.imshow(canvas)

    err = latents_upscaled - latent_canvas_test
    print(err.square().mean().sqrt())
