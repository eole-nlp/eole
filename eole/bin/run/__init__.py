from eole.bin import BaseBin
from eole.config.cli import add_model
from eole.config import get_non_default_values
from argparse import ArgumentParser
import os
import yaml


class RunBin(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--config",
            "-config",
            "-c",
            required=cls.require_config,
            help="Path of main YAML config file.",
        )
        add_model(parser, cls.config_class)

    @classmethod
    def build_config(cls, args):
        if args.config is not None:
            with open(args.config) as f:
                config_dict = yaml.safe_load(os.path.expandvars(f.read()))
        else:
            config_dict = {}

        # update config with non default values from command line args
        _parser = ArgumentParser()
        add_model(_parser, cls.config_class)
        defaults = vars(_parser.parse_args([]))
        stuff_to_update = get_non_default_values(args, defaults)
        config_dict.update(stuff_to_update)

        # pop extra fields added by argparse, not supported in pydantic configs
        if "bin" in config_dict.keys():
            config_dict.pop("bin")
        if "config" in config_dict.keys():
            config_dict.pop("config")

        config = cls.config_class(**config_dict)
        return config

    @classmethod
    def run(cls, args):
        raise NotImplementedError
