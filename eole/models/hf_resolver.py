import fnmatch
import json
import os
from glob import glob

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from eole.config.run import TrainConfig
from eole.utils.logging import logger


def is_hf_model_id(model_path):
    """Return True if model_path should be treated as a Hugging Face model ID.

    A path is treated as an HF model ID when it is relative, contains a slash,
    and does not exist on the local filesystem. This avoids misclassifying
    absolute or existing local model directories as Hub IDs.
    """
    s = str(model_path)
    if os.path.isabs(s) or s.startswith("."):
        return False
    return "/" in s and not os.path.exists(s)


def hf_cache_dir(model_id):
    slug = model_id.replace("/", "--").replace("\\", "--")
    return os.path.join(os.path.expanduser("~"), ".cache", "eole", "hf", slug)


def _repo_files_look_like_eole_model(repo_files):
    return (
        "config.json" in repo_files
        and "vocab.json" in repo_files
        and any(fnmatch.fnmatch(filename, "model.*.safetensors") for filename in repo_files)
    )


def _config_file_is_loadable_eole_model(config_path):
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        config["data"] = {}
        return TrainConfig(**config).model is not None
    except Exception as exc:
        logger.debug("Invalid EOLE config at %s: %s", config_path, exc)
        return False


def _local_dir_is_loadable_eole_model(model_dir):
    return (
        os.path.exists(os.path.join(model_dir, "vocab.json"))
        and bool(glob(os.path.join(model_dir, "model.*.safetensors")))
        and _config_file_is_loadable_eole_model(os.path.join(model_dir, "config.json"))
    )


def _remote_config_is_eole_model(repo_id, token, cache_dir):
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token, local_dir=cache_dir)
    return _config_file_is_loadable_eole_model(config_path)


def resolve_preconverted_eole_hf_repo(model_name, token=None):
    if not model_name or not is_hf_model_id(model_name):
        return None

    cache_dir = hf_cache_dir(model_name)
    if _local_dir_is_loadable_eole_model(cache_dir):
        return cache_dir

    try:
        repo_files = list_repo_files(repo_id=model_name, token=token)
    except Exception as exc:
        logger.debug("Could not inspect Hugging Face repo %s: %s", model_name, exc)
        return None

    if not _repo_files_look_like_eole_model(repo_files):
        return None

    try:
        if not _remote_config_is_eole_model(model_name, token, cache_dir):
            return None
    except Exception as exc:
        if _local_dir_is_loadable_eole_model(cache_dir):
            return cache_dir
        raise RuntimeError(
            f"Hugging Face repo {model_name} looks like an EOLE model but config validation failed."
        ) from exc

    logger.info("Downloading pre-converted EOLE model %s to %s", model_name, cache_dir)
    try:
        snapshot_download(repo_id=model_name, token=token, local_dir=cache_dir)
    except Exception as exc:
        if _local_dir_is_loadable_eole_model(cache_dir):
            return cache_dir
        raise RuntimeError(f"Hugging Face repo {model_name} looks like an EOLE model but download failed.") from exc

    if not _local_dir_is_loadable_eole_model(cache_dir):
        raise RuntimeError(f"Downloaded Hugging Face repo {model_name} is not a loadable EOLE model directory.")
    return cache_dir
