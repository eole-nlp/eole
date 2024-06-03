# some utilitary functions to facilitate cli tests
# and transition to config paradigm
import ast
import types
from typing import Union


def is_potentially_bool(field_type):
    if field_type is None:
        return False
    return bool in getattr(field_type, "__args__", [field_type])


def is_list(field_type):
    """This is very dirty and should not be kept in the future."""
    if field_type is None or isinstance(field_type, type(None)):
        return False
    # cases like "int | None" are types.UnionType instead of typing.Union
    elif isinstance(field_type, types.UnionType):
        return False
    elif getattr(field_type, "__origin__", None) == Union:
        return any([is_list(item) for item in field_type.__args__])
    else:
        return getattr(field_type, "_name", None) == "List"


def parse_type_annotation(annotation: str):
    try:
        # Parse the type annotation using ast
        tree = ast.parse(annotation, mode="eval")

        # Convert ast expression to a callable type
        return eval(compile(tree, filename="", mode="eval"))
    except (SyntaxError, TypeError):
        # Handle parsing errors
        return None


# test workaround to facilitate migration/testing
def add_model(parser, model):
    "Add Pydantic model to an ArgumentParser"
    fields = model.__fields__
    for name, field in fields.items():
        # special case for bool flags
        if is_potentially_bool(field.annotation):
            parser.add_argument(
                f"--{name}",
                f"-{name}",
                dest=name,
                action="store_true",
                default=field.default,
                help=field.description,
            )
        # we might also need one for nargs+
        elif is_list(field.annotation):
            parser.add_argument(
                f"--{name}",
                f"-{name}",
                dest=name,
                type=parse_type_annotation(field.annotation),
                nargs="+",
                default=field.default,
                help=field.description,
            )
        else:
            parser.add_argument(
                f"--{name}",
                f"-{name}",
                dest=name,
                type=parse_type_annotation(field.annotation),
                default=field.default,
                help=field.description
                # help=field.field_info.description,
            )
