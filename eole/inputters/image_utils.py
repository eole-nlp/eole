import numpy as np
from PIL import Image
from typing import Tuple

"""
Most of this code is borrowed from:
https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/multimodal.py
"""

# This is strictly restricted to pixtral-12b, extending to other models will require some mapping
DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)  # RGB
DATASET_STD = (0.26862954, 0.26130258, 0.27577711)  # RGB


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert a PIL image to RGB.
    We ensure transparent background becomes white.
    """
    if image.mode == "RGB":
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    white_bg: Image.Image = Image.new("RGBA", image.size, "WHITE")
    white_bg.paste(image, (0, 0), image)
    return white_bg.convert("RGB")


def normalize(np_image, mean, std):
    # np_image = np_image / 255.0
    assert len(np_image.shape) == 3, f"{np_image.shape=}"
    assert np_image.shape[2] == len(mean) == len(std), f"{np_image.shape=}, {mean=}, {std=}"
    mean = np.array(mean, dtype=np_image.dtype)
    std = np.array(std, dtype=np_image.dtype)
    image = (np_image - mean) / std
    return image.transpose(2, 0, 1)


def transform_image(image: Image.Image, new_size: Tuple[int, int]) -> np.ndarray:
    # resize PIL image directly
    resized_image = image.resize(new_size, resample=3, reducing_gap=None)  # hardcoded kwargs, reproducing HF
    np_image = np.array(_convert_to_rgb(resized_image), dtype=np.uint8)
    # rescale (dtype shennanigans to match HF processing)
    np_image = (np_image.astype(np.float64) * 1 / 255).astype(np.float32)
    return normalize(np_image, DATASET_MEAN, DATASET_STD)


def image_to_num_tokens(img, image_size=1024, image_patch_size=16):
    w, h = img.size
    ratio = max(h / image_size, w / image_size)
    if ratio > 1:
        w = round(w / ratio)
        h = round(h / ratio)
    width_tokens = (w - 1) // image_patch_size + 1
    height_tokens = (h - 1) // image_patch_size + 1
    return width_tokens, height_tokens


def process_image(image_path, image_size=1024, image_patch_size=16):
    image = Image.open(image_path)
    w, h = image_to_num_tokens(image, image_size=image_size, image_patch_size=image_patch_size)
    new_image_size = (w * image_patch_size, h * image_patch_size)
    # TODO retrieve from model config / vocab / tokenizer
    image_tokens = (["[IMG]"] * w + ["[IMG_BREAK]"]) * h
    image_tokens[-1] = "[IMG_END]"
    processed_image = transform_image(image, new_image_size)
    return {"image": processed_image, "tokens": image_tokens}
