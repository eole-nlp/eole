import os
import importlib
import inspect


class BaseBin(object):
    """
    Abstract base class for tools.
    """

    @classmethod
    def add_args(cls, parser):
        raise NotImplementedError

    @classmethod
    def run(cls, parser):
        raise NotImplementedError


AVAILABLE_BINS = {}


def register_bin(name):
    """Wrapper to automatically register available tools."""

    def register_bin_cls(cls):
        # patch to ignore standalone imports to prevent conflicts (e.g. in build docs)
        if not (cls.__module__.startswith("eole")):
            return cls
        # bin_type is retrieved from bin parent dirname
        bin_type = os.path.basename(os.path.dirname(inspect.stack()[1].filename))
        if bin_type in AVAILABLE_BINS:
            if (
                name in AVAILABLE_BINS[bin_type]
                and cls != AVAILABLE_BINS[bin_type][name]
            ):
                raise ValueError(
                    f'Name "{name}" registered twice for different classes: \n'
                    f"{AVAILABLE_BINS[bin_type][name]} != {cls}"
                )
            else:
                AVAILABLE_BINS[bin_type][name] = cls
        else:
            AVAILABLE_BINS[bin_type] = {name: cls}
        return cls

    return register_bin_cls


# Auto import python files in all subdirectories
bin_dir = os.path.dirname(__file__)
for dir in os.listdir(bin_dir):
    abs_dir = os.path.join(os.path.dirname(__file__), dir)
    if os.path.isdir(abs_dir):
        for file in os.listdir(abs_dir):
            _path = os.path.join(abs_dir, file)
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(_path))
            ):
                file_name = file[: file.find(".py")] if file.endswith(".py") else file
                module = importlib.import_module(f"eole.bin.{dir}." + file_name)
