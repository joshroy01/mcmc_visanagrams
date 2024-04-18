from typing import TYPE_CHECKING, Dict, Tuple, Any, Optional, List
import json
from pathlib import Path

import torch
from torch.nn.functional import interpolate

if TYPE_CHECKING:
    from mcmc_visanagrams.views.view_base import BaseView


def make_canvas(latents,
                canvas_size,
                sizes,
                in_channels=3,
                base_size=64,
                views: Optional[List['BaseView']] = None,
                do_edge_interpolation: bool = False):
    # Make a canvas from different latents
    canvas_count = torch.zeros(canvas_size, canvas_size).to(latents.device)
    canvas_latent = torch.zeros(1, in_channels, canvas_size, canvas_size,
                                dtype=latents.dtype).to(latents.device)

    for i, (size, latent) in enumerate(zip(sizes, latents)):
        latent = latent[None]
        sf, x_start, y_start = size
        size = min(canvas_size - x_start, base_size) * sf

        latent_expand = interpolate(latent, (size, size), mode='nearest')

        if views:
            latent_expand = views[i].inverse_view(latent_expand)

        weight = 1 / (sf**2)
        if do_edge_interpolation:
            coords = torch.linspace(-1, 1, size).to(latents.device)
            XX, YY = torch.meshgrid(coords, coords)
            dist_from_edge = torch.sqrt(torch.min((1 - torch.abs(XX))**2, (1 - torch.abs(YY))**2))
            dist_from_edge = torch.minimum(
                torch.tensor(6 / 32), dist_from_edge
            )  # only fall off along the outer edge, to avoid sharp lines at edge of each contributing image
            dist_from_edge = dist_from_edge / torch.max(dist_from_edge)
            weight = weight * dist_from_edge
        weight = weight + 1e-6

        canvas_latent[:, :, y_start:y_start + size, x_start:x_start +
                      size] = weight * latent_expand + canvas_latent[:, :, y_start:y_start + size,
                                                                     x_start:x_start + size]
        canvas_count[y_start:y_start + size, x_start:x_start +
                     size] = canvas_count[y_start:y_start + size, x_start:x_start + size] + weight

    canvas_latent = canvas_latent / (canvas_count[None, None] + 1e-10)
    canvas_latent = torch.nan_to_num(canvas_latent)
    return canvas_latent


def make_canvas_stage_2(latents,
                        canvas_size,
                        sizes,
                        in_channels=3,
                        views: Optional[List['BaseView']] = None,
                        do_edge_interpolation: bool = True,
                        base_size: int = 128):
    """Wrapper for make_canvas that uses target_size=128"""
    # TODO: Add some error checking here to debug stage 2 issues?
    return make_canvas(latents,
                       canvas_size,
                       sizes,
                       in_channels=in_channels,
                       views=views,
                       do_edge_interpolation=do_edge_interpolation,
                       base_size=base_size)


def save_model_spec(model_spec: Dict[str, str], dir_path: Path):
    output_path = dir_path / "model_spec.json"
    with output_path.open('w') as f:
        json.dump(model_spec, f, indent=4)
    print("Saved model specification dictionary to:", output_path.as_posix())


def load_model_spec(trial_root_dir: Path) -> Dict[str, str]:
    spec_path = trial_root_dir / "model_spec.json"
    with spec_path.open('r') as f:
        spec = json.load(f)
    print("Loaded model specification dictionary from:", spec_path.as_posix())
    return spec


def save_context(context: Dict[Tuple, Dict[str, Any]], path: Path):
    # If the path doesn't specify the filename of the context dictionary, then add it.
    if not path.suffix:
        path = path / "context.json"

    # Have to convert to JSON-serializable dictionary. The only thing that is an issue is that the
    # top-level key is a tuple which isn't JSON-serializable. So we convert it to a string.
    context_dict = {}
    for key, value in context.items():
        context_dict[str(key)] = value

    with path.open('w') as f:
        json.dump(context_dict, f, indent=4)

    print("Saved context to:", path.as_posix())


def load_context(path: Path) -> Dict[Tuple, Dict[str, Any]]:
    # If the path doesn't specify the filename of the context dictionary, then add it.
    if not path.suffix:
        path = path / "context.json"

    with path.open('r') as f:
        context_dict = json.load(f)

    # Convert the string key back to a tuple
    context = {}
    for key, value in context_dict.items():
        # Do a modicum of input validation since eval is scary.
        if not key.startswith('(') or not key.endswith(')'):
            raise ValueError(f'Invalid key: {key}, expected tuple')
        context[eval(key)] = value

    print("Loaded context from:", path.as_posix())

    return context
