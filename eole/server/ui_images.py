"""Bounded Chat image inputs for the Qwen vision adapters."""

from io import BytesIO
import json
import os
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field

from eole.server.ui_ocr import decode_data, open_image


class ChatImage(BaseModel):
    name: str = Field(max_length=200)
    data: str = Field(max_length=4000000)


def image_capabilities(model):
    config = getattr(getattr(model, "engine", None), "config", None)
    architecture = getattr(config, "model", None)
    if architecture is None:
        try:
            model_path = getattr(model, "local_path", None) or model.model_path
            path = Path(os.path.expandvars(model_path))
            if not path.is_absolute() and getattr(model, "models_root", None):
                path = Path(os.path.expandvars(model.models_root)) / path
            architecture = json.loads((path / "config.json").read_text()).get("model", {})
        except (AttributeError, OSError, ValueError, TypeError):
            architecture = {}
    adapter = architecture.get("adapter") if isinstance(architecture, dict) else getattr(architecture, "adapter", None)
    settings = getattr(model, "pre_config", {})
    data_type = getattr(config, "data_type", settings.get("data_type", "text"))
    return adapter in ("qwen3vl", "qwen3_5vl") and data_type == "image"


def chat_prompt(model, messages, thinking):
    """Return native input and exact expanded text for the context-budget check.

    Insert attachment sentinels as template text, expand them through the native
    image adapter, and escape literal braces for ImageTextCorpus formatting.
    """
    images, rendered = {}, []
    for message in messages:
        item = message.model_dump(exclude_none=True, exclude={"images"})
        if message.images:
            if message.role != "user":
                raise ValueError("Images can only be attached to user messages")
            content = []
            for attachment in message.images:
                if len(images) >= 4:
                    raise ValueError("At most four images are supported per conversation request")
                raw = decode_data(attachment.data, 3000000)
                with Image.open(BytesIO(raw)) as header:
                    if header.format not in ("JPEG", "PNG", "WEBP"):
                        raise ValueError("Chat images must be JPEG, PNG or WebP")
                image = open_image(raw)
                image.thumbnail((1024, 1024))
                key = f"eole_image_{len(images)}"
                images[key] = image
                # A unique textual marker keeps user-supplied vision tokens
                # from stealing an attachment's position in the rendered prompt.
                content.append({"type": "text", "text": f"EOLEIMAGEPLACEHOLDER{len(images) - 1}END"})
            content.append({"type": "text", "text": message.content})
            item["content"] = content
        rendered.append(item)
    if images and not image_capabilities(model):
        raise ValueError("Select an image-capable Qwen model configured with data_type: image")
    text = model.apply_chat_template(rendered, enable_thinking=thinking)
    if getattr(model.engine.config, "data_type", "text") != "image":
        return text, text
    native = text.replace("{", "{{").replace("}", "}}")
    expanded = text
    if images:
        from eole.inputters.image_utils import process_image

        config = model.engine.config.model
        encoder = config.encoder
        for index, (key, image) in enumerate(images.items()):
            marker = f"EOLEIMAGEPLACEHOLDER{index}END"
            if text.count(marker) != 1:
                raise ValueError("Image marker was altered by the chat template or appears in message text")
            processed = process_image(
                image,
                adapter=config.adapter,
                image_patch_size=encoder.patch_size * config.spatial_merge_size,
                image_size=encoder.image_size,
            )
            native = native.replace(marker, "{" + key + "}")
            expanded = expanded.replace(marker, "".join(processed["tokens"]))
    return {"text": native, "images": images}, expanded
