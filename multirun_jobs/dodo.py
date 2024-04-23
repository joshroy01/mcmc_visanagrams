from pathlib import Path

import yaml

from mcmc_visanagrams.runners.config import Config, OUTPUT_ROOT_PATH
from mcmc_visanagrams.runners.full_pipeline import run_full_pipeline
from mcmc_visanagrams.runners.multirun.multirun import generate_all_configs, populate_config_dict

ROOT = Path(__file__).resolve().parent
CONFIG_BASE_PATH = ROOT / "config_base.yml"
CONFIG_VARIADIC_PATH = ROOT / "config_variadic.yml"

MASTER_LIST_PATH = ROOT / "multirun_config_paths.txt"


def read_base_config():
    with CONFIG_BASE_PATH.open("r") as f:
        return yaml.safe_load(f)


def read_variadic_config():
    with CONFIG_VARIADIC_PATH.open("r") as f:
        return yaml.safe_load(f)


def populate_and_save_config(base_config, config_overrides, output_path):
    config = populate_config_dict(base_config, config_overrides)
    with output_path.open('w') as f:
        yaml.safe_dump(config, f)


def generate_and_save_config_paths():
    base_config = read_base_config()
    variadic_config = read_variadic_config()
    # Get the output path of all configs.
    all_configs = generate_all_configs(base_config, variadic_config["jobs"])

    # Write the configs.
    all_paths = []
    for config in all_configs:
        direc = OUTPUT_ROOT_PATH / config["trial_name"]
        direc.mkdir(parents=True, exist_ok=True)
        path = direc / "config.yaml"

        with path.open('w') as f:
            yaml.safe_dump(config, f)

        all_paths.append(path)

    # Write the file containing all config paths.
    with MASTER_LIST_PATH.open('w') as f:
        for path in all_paths:
            f.write(f"{path}\n")


# TODO: I should really make the generation of the master list a task that depends on the base
# configuration template and the variadic configuration(s). Then, creation of each config file
# should be a task. That way, the config file can be a dependency of the task that runs the
# pipeline.
# - This is likely the most "doit" way of organizing this.


def task_multirun():
    generate_and_save_config_paths()

    with MASTER_LIST_PATH.open('r') as f:
        master_list = f.readlines()
    all_config_paths = [Path(p.strip()) for p in master_list]

    for i, config_path in enumerate(all_config_paths):
        config_root = Config(config_path)
        for j, seed in enumerate([0, 90210, 8675309]):
            config = config_root.new_config_from_seed(seed)

            target = config.stage_2_output_path / "report.pdf"

            # TODO: If I want to get fancy, I could define sub-tasks for both stage 1 and stage 2 of the
            # pipeline with each target being the respective report.pdf.
            yield {
                'basename': f"multirun_job_{i}_seed{seed}",
                'doc': f"Run the full pipeline for config {config_path} and seed {config.seed}",
                'actions': [(run_full_pipeline, [config])],
                # Can't add config_path to the file_dep because it will trigger the task every time it's
                # run. Instead, we add the master list file as a dependency and check the target report.
                'file_dep': [MASTER_LIST_PATH],
                'targets': [target]
            }
