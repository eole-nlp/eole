import torch
import numpy as np
from PIL import Image
import PIL
from typing import Tuple, Optional, Union
from enum import Enum
from collections.abc import Collection
from eole.utils.logging import logger

"""
Most of this code is borrowed from:
https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/multimodal.py
"""

# This is strictly restricted to pixtral-12b, extending to other models will require some mapping
DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)  # RGB
DATASET_STD = (0.26862954, 0.26130258, 0.27577711)  # RGB


class ChannelDimension(Enum):
    FIRST = "channels_first"
    LAST = "channels_last"


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


def normalize_llava(np_image, mean, std):
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
    return normalize_llava(np_image, DATASET_MEAN, DATASET_STD)


def image_to_num_tokens(img, image_size=1024, image_patch_size=16):
    w, h = img.size
    ratio = max(h / image_size, w / image_size)
    if ratio > 1:
        w = round(w / ratio)
        h = round(h / ratio)
    width_tokens = (w - 1) // image_patch_size + 1
    height_tokens = (h - 1) // image_patch_size + 1
    return width_tokens, height_tokens


def _rescale_for_pil_conversion(image):
    """
    Detects whether or not the image needs to be rescaled before being converted to a PIL image.

    The assumption is that if the image is of type `np.float` and all values are between 0 and 1, it needs to be
    rescaled.
    """
    if image.dtype == np.uint8:
        do_rescale = False
    elif np.allclose(image, image.astype(int)):
        if np.all(0 <= image) and np.all(image <= 255):
            do_rescale = False
        else:
            raise ValueError(
                "The image to be converted to a PIL image contains values outside the range [0, 255], "
                f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
            )
    elif np.all(0 <= image) and np.all(image <= 1):
        do_rescale = True
    else:
        raise ValueError(
            "The image to be converted to a PIL image contains values outside the range [0, 1], "
            f"got [{image.min()}, {image.max()}] which cannot be converted to uint8."
        )
    return do_rescale


def to_pil_image(
    image: Union[np.ndarray, "PIL.Image.Image", "torch.Tensor"],
    do_rescale: Optional[bool] = None,
    image_mode: Optional[str] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> "PIL.Image.Image":
    """
    Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
    needed.

    Args:
        image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
            The image to convert to the `PIL.Image` format.
        do_rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will default
            to `True` if the image type is a floating type and casting to `int` would result in a loss of precision,
            and `False` otherwise.
        image_mode (`str`, *optional*):
            The mode to use for the PIL image. If unset, will use the default mode for the input image type.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `PIL.Image.Image`: The converted image.
    """
    # requires_backends(to_pil_image, ["vision"])

    if isinstance(image, PIL.Image.Image):
        return image

    # Convert all tensors to numpy arrays before converting to PIL image
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        raise ValueError(f"Input image type not supported: {type(image)}")

    # If the channel has been moved to first dim, we put it back at the end.
    image = to_channel_dimension_format(image, ChannelDimension.LAST, input_data_format)

    # If there is a single channel, we squeeze it, as otherwise PIL can't handle it.
    image = np.squeeze(image, axis=-1) if image.shape[-1] == 1 else image

    # PIL.Image can only store uint8 values so we rescale the image to be between 0 and 255 if needed.
    do_rescale = _rescale_for_pil_conversion(image) if do_rescale is None else do_rescale

    if do_rescale:
        image = rescale(image, 255)

    image = image.astype(np.uint8)
    return PIL.Image.fromarray(image, mode=image_mode)


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


def resize(
    image: np.ndarray,
    size: Tuple[int, int],
    resample: "Image.Resampling" = None,
    reducing_gap: Optional[int] = None,
    data_format: Optional[ChannelDimension] = None,
    return_numpy: bool = True,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Resizes `image` to `(height, width)` specified by `size` using the PIL library.

    Args:
        image (`np.ndarray`):
            The image to resize.
        size (`Tuple[int, int]`):
            The size to use for resizing the image.
        resample (`int`, *optional*, defaults to `Image.Resampling.BILINEAR`):
            The filter to user for resampling.
        reducing_gap (`int`, *optional*):
            Apply optimization by resizing the image in two steps. The bigger `reducing_gap`, the closer the result to
            the fair resampling. See corresponding Pillow documentation for more details.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        return_numpy (`bool`, *optional*, defaults to `True`):
            Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is
            returned.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `np.ndarray`: The resized image.
    """
    # requires_backends(resize, ["vision"])

    resample = resample if resample is not None else Image.Resampling.BILINEAR

    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    do_rescale = False
    if not isinstance(image, PIL.Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(image, do_rescale=do_rescale, input_data_format=input_data_format)
    height, width = size
    # PIL images are in the format (width, height)
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)

    if return_numpy:
        resized_image = np.array(resized_image)
        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        # The image is always in channels last format after converting from a PIL image
        resized_image = to_channel_dimension_format(resized_image, data_format, input_channel_dim=ChannelDimension.LAST)
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        resized_image = rescale(resized_image, 1 / 255) if do_rescale else resized_image
    return resized_image


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


def normalize_gemma(
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


def process_image(image_path, adapter="llava", image_size=1024, image_patch_size=16):
    if adapter == "llava":
        image = Image.open(image_path)
        w, h = image_to_num_tokens(image, image_size=image_size, image_patch_size=image_patch_size)
        new_image_size = (w * image_patch_size, h * image_patch_size)
        # TODO retrieve from model config / vocab / tokenizer
        image_tokens = (["[IMG]"] * w + ["[IMG_BREAK]"]) * h
        image_tokens[-1] = "[IMG_END]"
        processed_image = transform_image(image, new_image_size)
        return {"image": processed_image, "tokens": image_tokens}
    elif adapter == "gemma3":
        image = Image.open(image_path)
        image = _convert_to_rgb(image)
        image = resize(image=image, size=(896, 896), resample=2, input_data_format=ChannelDimension.LAST)
        image = rescale(image=image, scale=0.00392156862745098, input_data_format=ChannelDimension.LAST)  # 1/256
        image = normalize_gemma(
            image=image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], input_data_format=ChannelDimension.LAST
        )
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=ChannelDimension.LAST)
        # return image
        # TODO: make this configurable?
        image_tokens = "<start_of_image>" + "<image_soft_token>" * 256 + "<end_of_image>"
        return {"image": image, "tokens": image_tokens}
    else:
        raise ValueError("Unsupported Adapter type: {}".format(adapter))
