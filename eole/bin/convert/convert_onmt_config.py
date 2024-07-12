from eole.bin import BaseBin, register_bin
import yaml
from eole.config.run import TrainConfig
from eole.config.training import TrainingConfig
from eole.config.run import NestedAllTransformsConfig
from collections import defaultdict, OrderedDict
from eole.config.models import BaseModelConfig
from rich import print
from eole.config import reorder_fields


# to be extended
KEY_MAP = {
    "save_model": ["training", "model_path"],
    "encoder_type": [
        "model",
        "encoder",
        "encoder_type",
    ],  # will set config["model"]["encoder"]["encoder_type"]
    "decoder_type": ["model", "decoder", "decoder_type"],
    "enc_layers": ["model", "encoder", "layers"],
    "dec_layers": ["model", "decoder", "layers"],
    "heads": ["model", "heads"],
    "model_dtype": ["training", "compute_dtype"],
    "max_relative_positions": ["model", "embeddins", "n_positions"],
    "num_kv": ["model", "heads_kv"],
    "pos_ffn_activation_fn": ["model", "mlp_activation_fn"],
}


def custom_mapping(k, v, config):
    # to be extended
    match k, v:
        case "max_relative_positions", -1:
            config["model"]["embeddings"]["position_encoding_type"] = "Rotary"
        case "max_relative_positions", -2:
            config["model"]["embeddings"]["position_encoding_type"] = "Alibi"
        case "max_relative_positions", _ if v > 0:
            config["model"]["embeddings"]["positions_encoding_type"] = "Relative"
        case "max_relative_positions", 0:
            config["model"]["embeddings"]["position_encoding_type"] = "Absolute"


def set_nested_value(d, keys, value):
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _to_dict(d):
    if isinstance(d, defaultdict) or isinstance(d, OrderedDict):
        d = {k: _to_dict(v) for k, v in d.items()}
    return d


@register_bin(name="onmt_config")
class OnmtConfigConverter(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("input", help="Input OpenNMT-py yaml config to convert.")
        parser.add_argument("output", help="Output converted yaml EOLE config.")
        parser.add_argument(
            "-a",
            "--architecture",
            choices=["rnn", "cnn", "transformer", "transformer_lm"],
            help="Setting this field will ensure conversion will be relevant to the architecture.",
        )

    @classmethod
    def run(cls, args):
        with open(args.input) as f:
            data_in = yaml.safe_load(f)
        # 3 potential levels of nesting
        new_config = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        if "data" in data_in.keys():
            new_config["data"] = data_in["data"]
        if args.architecture is not None:
            new_config["model"]["architecture"] = args.architecture
        mapped = {"data"}
        # retrieve all transforms to populate only necessary transforms_configs
        all_transforms = set(data_in.get("transforms", []))
        for _, corpus in data_in.get("data", {}).items():
            all_transforms |= set(corpus.get("transforms", []))
        for k, v in data_in.items():
            if k in KEY_MAP.keys():
                set_nested_value(new_config, KEY_MAP[k], v)
                mapped.add(k)
            if k not in mapped:
                if k in TrainConfig.model_fields:
                    new_config[k] = v
                    mapped.add(k)
                elif k in TrainingConfig.model_fields:
                    new_config["training"][k] = v
                    mapped.add(k)
                elif k in BaseModelConfig.model_fields:
                    new_config["model"][k] = v
                    mapped.add(k)
                else:
                    for (
                        t_name,
                        t_field,
                    ) in NestedAllTransformsConfig.model_fields.items():
                        if k in t_field.default.model_fields:
                            if t_name in all_transforms:
                                try:
                                    # handle edge cases like {src,tgt}_onmttok_kwargs
                                    # str to dict conversion
                                    v = eval(v)
                                except Exception:
                                    pass
                                new_config["transforms_configs"][t_name][k] = v
                                mapped.add(k)
        unmapped = set(data_in.keys()) - mapped
        # reorder fields for readability
        new_config = reorder_fields(new_config)
        # convert to standard dict for proper yaml dump
        new_config = _to_dict(new_config)
        print(new_config)
        print("Saving converted config to:", args.output)
        with open(args.output, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        print("Remaining unmapped items:", unmapped)
        # test config validity
        new_config["data"] = {}
        try:
            TrainConfig(**new_config)
        except Exception as e:
            print(e)
        else:
            print("Converted config seems ok!")
