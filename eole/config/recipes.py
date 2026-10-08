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


class LiveCodeBenchConfig(Config):
    recipe_type: Literal["livecodebench"]
    upstream_dir: str
    output_dir: str
    release_version: Literal["release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6"]
    start_date: str | None = None
    end_date: str | None = None
    limit: int | None = Field(default=None, ge=1)
    base_url: str
    model_id: str
    max_tokens: int = Field(default=8192, ge=1)
    request_timeout: int = Field(default=1800, ge=1)
    num_process_evaluate: int = Field(default=2, ge=1)
    test_timeout: int = Field(default=6, ge=1)
