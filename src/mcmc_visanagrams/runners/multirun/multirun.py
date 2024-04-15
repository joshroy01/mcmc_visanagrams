from typing import Dict, Any, List
import copy

from mcmc_visanagrams.runners.config import Config
from mcmc_visanagrams.runners.full_pipeline import run_full_pipeline


def _recurse_update_list(base, override):
    for i, v in enumerate(override):
        if isinstance(v, dict):
            _recurs_update_dict(base[i], v)
        elif isinstance(v, list):
            _recurse_update_list(base[i], v)
        else:
            base[i] = v


def _recurs_update_dict(base, override):
    for k, v in override.items():
        if k == "trial_name":
            base[k] = f"{base[k]}/{v}"
        elif isinstance(v, dict):
            _recurs_update_dict(base[k], v)
        elif isinstance(v, list):
            _recurse_update_list(base[k], v)
        else:
            base[k] = v


def populate_config_dict(config_base: Dict[str, Any],
                         config_overrides: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config_base)
    _recurs_update_dict(config, config_overrides)
    return config


def generate_all_configs(config_base: Dict[str, Any],
                         config_overrides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    configs = []
    for override in config_overrides:
        configs.append(populate_config_dict(config_base, override))
    return configs


def run_pipeline_from_config_dict(config_dict: Dict[str, Any]):
    config = Config.from_dict(config_dict)
    run_full_pipeline(config)


if __name__ == "__main__":
    config_base = {"a": 1, "b": {"c": 2, "d": [3, 4, 5], "e": {"f": 4, "g": 5}}}

    config_overrides = {"a": 10, "b": {"c": 20, "d": [5, 6, 7], "e": {"f": 40}}}

    print(populate_config_dict(config_base, config_overrides))
    # Expected output:
    # {'a': 10, 'b': {'c': 20, 'd': 3, 'e': {'f': 40, 'g': 5}}}
