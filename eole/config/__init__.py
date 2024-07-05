import os
from collections import OrderedDict
from eole.config.config import Config
from eole.utils.logging import logger


# default EOLE_MODEL_DIR
if os.environ.get("EOLE_MODEL_DIR", None) is None:
    os.environ["EOLE_MODEL_DIR"] = os.getcwd()


def calculate_depth(value, current_depth=0):
    if isinstance(value, (dict, Config)):
        if isinstance(value, Config):
            value = value.__dict__
        return (
            max(calculate_depth(v, current_depth + 1) for v in value.values())
            if value
            else current_depth
        )
    return current_depth


def reorder_fields(fields):
    """
    Put non nested fields before nested ones in config json dump,
    for better readability.
    """
    ordered_fields = OrderedDict()
    non_nested_fields = []
    nested_fields = []

    for field, value in fields.items():
        if isinstance(value, dict):
            nested_fields.append((field, value))
        else:
            non_nested_fields.append((field, value))

    # Add non-nested fields first
    for field, value in non_nested_fields:
        ordered_fields[field] = value

    # Calculate depths and sort nested fields
    nested_fields.sort(key=lambda x: calculate_depth(x[1]))

    # Add nested fields
    for field, value in nested_fields:
        ordered_fields[field] = reorder_fields(value)

    return ordered_fields


def recursive_model_fields_set(model):
    fields = {}
    if isinstance(model, dict):
        fields_to_check = model.keys()
    else:
        # we need to explicitly handle "architecture" related fields for now
        # to properly handle discriminated config instanciation
        discriminator_fields = {
            "architecture",
            "encoder_type",
            "decoder_type",
        }.intersection(set(model.__fields__.keys()))
        nested_models = {
            key for key, value in model.__dict__.items() if isinstance(value, Config)
        }
        fields_to_check = model.model_fields_set | discriminator_fields | nested_models
    for field in fields_to_check:
        if isinstance(model, dict):
            field_value = model.get(field, None)
        else:
            field_value = getattr(model, field, None)
        if isinstance(field_value, Config) or isinstance(field_value, dict):
            _fields = recursive_model_fields_set(field_value)
            if _fields != {}:
                fields[field] = _fields
        else:
            fields[field] = field_value
    return reorder_fields(fields)


def recursive_update_dict(_dict, new_dict, defaults):
    # patch to allow populating previously none keys
    # (e.g. data in finetuned HF converted model)
    if _dict is None:
        _dict = {}

    for k, v in new_dict.items():
        if k == "architecture":
            # keep info from checkpoint
            continue
        if isinstance(v, dict):
            _dict[k] = recursive_update_dict(_dict.get(k, {}), v, defaults.get(k, {}))
        else:
            default = defaults.get(k, None) if isinstance(defaults, dict) else None
            previous_v = _dict.get(k, default)
            if v != previous_v:
                logger.info(f"Option: {k}, value: {v}, overriding model: {previous_v}")
            _dict[k] = v
    return _dict


def get_non_default_values(parsed_args, defaults):
    non_default_values = {}
    # defaults = vars(parse_args())
    for key, value in vars(parsed_args).items():
        if value != defaults.get(key, None):
            non_default_values[key] = value
    return non_default_values
