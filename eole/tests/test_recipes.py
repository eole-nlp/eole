from argparse import ArgumentParser
from eole.config.run import TrainConfig, PredictConfig
from glob import glob
import os
import yaml

parser = ArgumentParser()

parser.add_argument("recipes_dir")

args = parser.parse_args()

configs_to_check = glob(os.path.join(args.recipes_dir, "*", "*.yaml"))

for config_path in configs_to_check:
    print(f"Checking: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    if "training" in config_dict.keys() or "model" in config_dict.keys():
        # must be a TrainConfig
        try:
            # pop data to prevent missing data error
            config_dict["data"] = {}
            train_config = TrainConfig(**config_dict)
        except Exception as e:
            raise e
    elif "models" in config_dict.keys():
        # server example config, ignore for now
        pass
    else:
        # must be a PredictConfig
        try:
            predict_config = PredictConfig(**config_dict)
        except Exception as e:
            raise e
