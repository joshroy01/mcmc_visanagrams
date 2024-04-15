from pathlib import Path

import yaml

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

        # Make the trial output path and save the config file in it.
        self.trial_output_path.mkdir(parents=True, exist_ok=True)
        self.stage_1_output_path.mkdir(exist_ok=True)
        self.stage_2_output_path.mkdir(exist_ok=True)
        with (self.trial_output_path / "config.yaml").open("w") as f:
            yaml.safe_dump(config, f)

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
        for context in config["context_list"]:
            size = tuple(context.pop("size"))
            context_list.append(Context(size=size, **context))
        return context_list
