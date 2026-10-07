"""Schemas for recipe orchestration settings, distinct from model run configs."""

from typing import Literal

from pydantic import Field

from eole.config.config import Config


class Qwen38ValidationConfig(Config):
    recipe_type: Literal["qwen38_validation"]
    model_path: str
    output_dir: str
    baseline_config: str
    mtp_config: str
    server_config: str
    model_id: str
    port: int = Field(default=5010, ge=1, le=65535)
    claude: bool = False
