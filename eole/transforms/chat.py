"""Chat-template transforms for instruction and chat-style datasets."""

from typing import Any, Dict, List

from jinja2 import Environment, StrictUndefined
from pydantic import Field

from eole.transforms import register_transform
from .transform import Transform, TransformConfig


class ChatConfig(TransformConfig):
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Messages to render into a chat prompt.")
    add_generation_prompt: bool = Field(default=True, description="Append the assistant generation prompt.")
    chat_template_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Extra variables to pass to the chat template renderer."
    )
    target: Dict[str, Any] = Field(default_factory=dict, description="Optional target normalization settings.")


@register_transform(name="chat")
class ChatTransform(Transform):
    """Render raw examples with the converted model's chat template."""

    config_model = ChatConfig
    supports_dataset_overrides = True

    def _parse_config(self):
        self.template = None
        self.specials = {}
        self.global_config = None
        self.corpus_configs = {}
        self.environment = Environment(undefined=StrictUndefined)
        self.environment.globals["raise_exception"] = self._raise_exception
        self.environment.globals["strftime_now"] = self._strftime_now

    def warm_up(self, vocabs=None):
        super().warm_up(vocabs)
        self.specials = vocabs.get("specials", {}) if vocabs is not None else {}
        inference_config = getattr(self.full_config, "inference", None)
        self.template = getattr(inference_config, "chat_template", None) if inference_config is not None else None
        if not self.template:
            raise ValueError("chat transform requires inference.chat_template from the converted EOLE model config.")
        self.compiled_template = self.environment.from_string(self.template)
        self.global_config = self.config
        self.corpus_configs = self._build_corpus_configs()
        self._validate_messages_configured()

    def _build_corpus_configs(self):
        corpus_configs = {}
        if getattr(self.full_config, "data", None) is None:
            return corpus_configs
        global_config = self.global_config.model_dump()
        for corpus_name, corpus in self.full_config.data.items():
            overrides = getattr(corpus, "transforms_configs", None) or {}
            chat_override = overrides.get(self.name)
            if chat_override is None:
                continue
            merged = {**global_config, **chat_override}
            corpus_configs[corpus_name] = ChatConfig.model_validate(merged)
        return corpus_configs

    def _chat_corpora_without_messages(self):
        missing = []
        if getattr(self.full_config, "data", None) is None:
            return missing
        for corpus_name, corpus in self.full_config.data.items():
            if self.name in (corpus.transforms or []) and corpus_name not in self.corpus_configs:
                missing.append(corpus_name)
        return missing

    def _validate_messages_configured(self):
        if self.global_config.messages:
            return
        missing = self._chat_corpora_without_messages()
        if missing:
            corpus_list = ", ".join(missing)
            raise ValueError(
                "chat transform requires global messages or dataset-level chat.messages overrides "
                f"for each chat corpus. Missing messages for: {corpus_list}."
            )

    @staticmethod
    def _raise_exception(message):
        raise ValueError(message)

    @staticmethod
    def _strftime_now(format_string):
        from datetime import datetime

        return datetime.now().strftime(format_string)

    @staticmethod
    def _format_value(value, fields):
        if isinstance(value, str):
            return value.format(**fields)
        if isinstance(value, list):
            return [ChatTransform._format_value(item, fields) for item in value]
        if isinstance(value, dict):
            return {key: ChatTransform._format_value(item, fields) for key, item in value.items()}
        return value

    def _render_messages(self, example, config):
        fields = dict(example)
        messages = [self._format_value(message, fields) for message in config.messages]
        render_kwargs = {
            "messages": messages,
            "add_generation_prompt": config.add_generation_prompt,
            "tools": None,
            "bos_token": self.specials.get("bos_token"),
            "eos_token": self.specials.get("eos_token"),
            "pad_token": self.specials.get("pad_token"),
            "unk_token": self.specials.get("unk_token"),
            **config.chat_template_kwargs,
        }
        return self.compiled_template.render(**render_kwargs)

    @staticmethod
    def _normalize_target(text, config):
        if text is None:
            return None
        if config.target.get("strip_commas", False):
            text = text.replace(",", "")
        return text

    def apply(self, example, is_train=False, stats=None, **kwargs):
        assert isinstance(example["src"], str), "ChatTransform requires a string source as input"
        config = self.corpus_configs.get(kwargs.get("corpus_name"), self.global_config)
        if not config.messages:
            raise ValueError(
                "chat transform has no messages for this corpus. Configure global chat.messages or a "
                "dataset-level chat.messages override."
            )
        example["src"] = self._render_messages(example, config)
        if example.get("tgt") is not None:
            example["tgt"] = self._normalize_target(example["tgt"], config)
            if "raw_tgt" in example:
                example["raw_tgt"] = self._normalize_target(example["raw_tgt"], config)
        return example
