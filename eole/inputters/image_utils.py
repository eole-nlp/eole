import numpy as np
from PIL import Image, ImageOps
from typing import Optional, Union
from enum import Enum
from collections.abc import Collection
from eole.utils.logging import logger


DATASET_MEAN = {
    "llava": (0.48145466, 0.4578275, 0.40821073),
    "gemma3": (0.5, 0.5, 0.5),
    "deepseekocr": (0.5, 0.5, 0.5),
    "hunyuanocr": (0.48145466, 0.4578275, 0.40821073),
}
DATASET_STD = {
    "llava": (0.26862954, 0.26130258, 0.27577711),
    "gemma3": (0.5, 0.5, 0.5),
    "deepseekocr": (0.5, 0.5, 0.5),
    "hunyuanocr": (0.26862954, 0.26130258, 0.27577711),
}
RESAMPLE = {
    "llava": Image.BICUBIC,
    "gemma3": Image.BILINEAR,
    "deepseekocr": Image.BILINEAR,
    "hunyuanocr": Image.BILINEAR,
}
SQUARE = {"llava": False, "gemma3": True, "deepseekocr": True, "hunyuanocr": False}


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


def image_to_num_tokens(img, image_size=0, image_patch_size=16, square=False):
    if square:  # Gemma3 (896x896) and Deepseekocr (1024x1024) require square fixed size
        return image_size // image_patch_size, image_size // image_patch_size
    w, h = img.size
    if image_size == 0 or image_size is None:
        image_size = max(w, h)
    ratio = max(h / image_size, w / image_size)
    if ratio > 1:
        w = round(w / ratio)
        h = round(h / ratio)
    width_tokens = (w - 1) // image_patch_size + 1
    height_tokens = (h - 1) // image_patch_size + 1
    # make numbers even to ensure batch even number of img tokens
    width_tokens = width_tokens + (width_tokens % 2)
    height_tokens = height_tokens + (height_tokens % 2)
    return width_tokens, height_tokens


class ChannelDimension(Enum):
    FIRST = "channels_first"
    LAST = "channels_last"


def infer_channel_dimension_format(
    image: np.ndarray, num_channels: Optional[Union[int, tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels and image.shape[last_dim] in num_channels:
        logger.warning(
            f"The channel dimension is ambiguous. Got image shape {image.shape}."
            f"Assuming channels are the first dimension."
        )
        return ChannelDimension.FIRST
    elif image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")


def rescale(
    image: np.ndarray,
    scale: float,
    data_format: Optional[ChannelDimension] = None,
    dtype: np.dtype = np.float32,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Rescales `image` by `scale`.

    Args:
        image (`np.ndarray`):
            The image to rescale.
        scale (`float`):
            The scale to use for rescaling the image.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        dtype (`np.dtype`, *optional*, defaults to `np.float32`):
            The dtype of the output image. Defaults to `np.float32`. Used for backwards compatibility with feature
            extractors.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The rescaled image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    rescaled_image = image.astype(np.float64) * scale  # Numpy type promotion has changed, so always upcast first
    if data_format is not None:
        rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)

    rescaled_image = rescaled_image.astype(dtype)  # Finally downcast to the desired dtype at the end

    return rescaled_image


def get_channel_dimension_axis(
    image: np.ndarray, input_data_format: Optional[Union[ChannelDimension, str]] = None
) -> int:
    """
    Returns the channel dimension axis of the image.

    Args:
        image (`np.ndarray`):
            The image to get the channel dimension axis of.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the image. If `None`, will infer the channel dimension from the image.

    Returns:
        The channel dimension axis of the image.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    if input_data_format == ChannelDimension.FIRST:
        return image.ndim - 3
    elif input_data_format == ChannelDimension.LAST:
        return image.ndim - 1
    raise ValueError(f"Unsupported data format: {input_data_format}")


def normalize(
    image: np.ndarray,
    mean: Union[float, Collection[float]],
    std: Union[float, Collection[float]],
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.

    image = (image - mean) / std

    Args:
        image (`np.ndarray`):
            The image to normalize.
        mean (`float` or `Collection[float]`):
            The mean to use for normalization.
        std (`float` or `Collection[float]`):
            The standard deviation to use for normalization.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
    num_channels = image.shape[channel_axis]

    # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
    # We preserve the original dtype if it is a float type to prevent upcasting float16.
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    if isinstance(mean, Collection):
        if len(mean) != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
    else:
        mean = [mean] * num_channels
    mean = np.array(mean, dtype=image.dtype)

    if isinstance(std, Collection):
        if len(std) != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
    else:
        std = [std] * num_channels
    std = np.array(std, dtype=image.dtype)

    if input_data_format == ChannelDimension.LAST:
        image = (image - mean) / std
    else:
        image = ((image.T - mean) / std).T

    image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
    return image


def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`numpy.ndarray`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`: The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))
    return image


def process_image(image_path, adapter="llava", image_patch_size=16, image_size=1024):
    if isinstance(image_path, Image.Image):
        PILimage = image_path
    else:
        PILimage = Image.open(image_path)

    PILimage = _convert_to_rgb(PILimage)

    if adapter in ["deepseekocr", "gemma3"]:
        # padding
        PILimage = ImageOps.pad(PILimage, (image_size, image_size), color=(127, 127, 127))
    else:
        # resize
        w, h = image_to_num_tokens(
            PILimage, image_size=image_size, image_patch_size=image_patch_size, square=SQUARE[adapter]
        )
        new_image_size = (w * image_patch_size, h * image_patch_size)
        PILimage = PILimage.resize(new_image_size, resample=RESAMPLE[adapter], reducing_gap=None)
    logger.info(f"Adapter is {adapter} - padded/resized PIL image is (w, h) {PILimage.size}")

    NPimage = np.array(PILimage, dtype=np.uint8)  # (H, W, C)
    NPimage = rescale(image=NPimage, scale=1 / 255, input_data_format=ChannelDimension.LAST)
    NPimage = normalize(
        image=NPimage,
        mean=DATASET_MEAN[adapter],
        std=DATASET_STD[adapter],
        input_data_format=ChannelDimension.LAST,
    )
    NPimage = to_channel_dimension_format(NPimage, ChannelDimension.FIRST, input_channel_dim=ChannelDimension.LAST)
    # NPimage is (C, H, W)

    if adapter == "llava":
        image_tokens = (["[IMG]"] * w + ["[IMG_BREAK]"]) * h
        image_tokens[-1] = "[IMG_END]"
    elif adapter == "gemma3":
        image_tokens = "<start_of_image>" + "<image_soft_token>" * 256 + "<end_of_image>"
    elif adapter == "deepseekocr":
        # sam patches are 16 (hardcoded in original implementation)
        # for a 1024x1024 image it means patches of 64x64
        # for the sam encoder it is 64x64 = 4096 patches
        # then goes to Clip 4096 / 16 = 256 tokens (16 patches x 16 patches)
        # but there is one token at the end of each line, and one more at the end of the pic
        # (16 + 1) x 16 + 1 = 273
        image_tokens = ["<image>"] * 273
    elif adapter == "hunyuanocr":
        image_tokens = (
            "<｜hy_place▁holder▁no▁100｜>"
            + "<｜hy_place▁holder▁no▁102｜>" * ((w + 1) * h + 2)
            + "<｜hy_place▁holder▁no▁101｜>"
        )
    else:
        raise ValueError("Unsupported Adapter type: {}".format(adapter))
    return {"image": NPimage, "tokens": image_tokens}
