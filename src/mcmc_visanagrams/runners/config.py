from pathlib import Path

import yaml
from copy import deepcopy

from mcmc_visanagrams.context import ContextList, Context

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
        self.stage_2_args = config["stage_2_args"]

    def new_config_from_seed(self, seed: int):
        config = deepcopy(self)

        config.seed = seed
        config._config_dict["seed"] = seed
        config.trial_output_path /= f"seed_{seed}"
        config.trial_output_path.mkdir(parents=True, exist_ok=True)
        config.seed = seed
        config.stage_1_output_path = config.trial_output_path / "stage_1"
        config.stage_2_output_path = config.trial_output_path / "stage_2"
        config.stage_1_output_path.mkdir(exist_ok=True)
        config.stage_2_output_path.mkdir(exist_ok=True)
        config.mkdirs()

        return config

    def mkdirs(self):
        # Make the trial output path and save the config file in it.
        self.trial_output_path.mkdir(parents=True, exist_ok=True)
        self.stage_1_output_path.mkdir(exist_ok=True)
        self.stage_2_output_path.mkdir(exist_ok=True)
        with (self.trial_output_path / "config.yaml").open("w") as f:
            yaml.safe_dump(self._config_dict, f)

    @staticmethod
    def from_dict(config_dict):
        output_dir = OUTPUT_ROOT_PATH / config_dict["trial_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "config.yaml"
        with output_path.open("w") as f:
            yaml.safe_dump(config_dict, f)
        return Config(output_path)

    def _context_list_from_config(self, config):
        context_list = ContextList()
        context_orig = deepcopy(config["context_list"])
        for context in context_orig:
            size = tuple(context.pop("size"))
            context_list.append(Context(size=size, **context))
        return context_list
