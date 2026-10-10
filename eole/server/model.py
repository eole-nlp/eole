"""Model loading, request batching, and inference configuration."""

import asyncio
import datetime
import gc
import json
import os
import time
from dataclasses import dataclass

import torch
import yaml
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from eole.inference_engine import InferenceEnginePY
from eole.config.run import PredictConfig
from eole.utils.logging import logger
from eole.server.utils import _normalize_developer_role, estimate_tokens


class Server(object):
    """
    Main server class to manage configuration, models and corresponding constraints.
    """

    def __init__(self):
        self.start_time = time.time()
        self.models = {}
        self.models_root = None

    def start(self, server_config_path):
        """
        Initialize the server with the given configuration.
        """
        with open(server_config_path) as f:
            server_config = yaml.safe_load(f)
        self.models_root = server_config["models_root"]
        for model in server_config["models"]:
            # instantiate models
            model_id = model["id"]
            model_path = model["path"]
            self.models[model_id] = Model(
                model_id=model_id,
                model_path=model_path,
                models_root=self.models_root,
                model_type=model.get("model_type", "default"),
                pre_config=model.get("config", {}),
            )
            if model.get("preload", False):
                self.models[model_id].load()

    def available_models(self):
        """
        Return a list of available models.
        """
        models = []
        for model_id, model in self.models.items():
            models.append({"id": model_id})
        return models

    async def maybe_load_model(self, model_id_to_load):
        """
        Very naive method to ensure a single model is loaded for now.
        """
        for model_id, model in self.models.items():
            if model_id != model_id_to_load:
                model.unload()


@dataclass
class QueuedRequest:
    inputs: any
    settings: dict
    is_chat: bool
    future: asyncio.Future
    timestamp: float


class Model(object):
    """
    Represents a single model in the server.
    """

    def __init__(
        self,
        model_id=None,
        model_path=None,
        preload=False,
        models_root=None,
        model_type=False,
        pre_config={},
    ):
        self.loaded = False
        self.engine = None
        self.model_id = model_id
        self.preload = preload
        self.models_root = models_root
        self.model_path = model_path
        self.local_path = None
        self.model_type = model_type
        self.pre_config = pre_config
        self.request_queue = asyncio.Queue()
        self.batch_size = pre_config.get("batch_size", 8)
        self.batch_timeout = pre_config.get("batch_timeout", 0.1)
        self.processing_task = None

    def get_config(self):
        """
        Instanciate the configuration for the model.
        """
        # transforms and inference settings are retrieved from the model config for now
        self.config = PredictConfig(
            src="dummy",
            model_path=self.local_path,
            # TODO improve this
            gpu_ranks=[0],
            world_size=1,
            **self.pre_config,
        )

    async def _process_batch(self, batch):
        """
        Process a batch of requests together.
        """
        try:
            # Helper function to make settings hashable via JSON
            def settings_key(settings, is_chat):
                """Create a hashable key from settings."""
                return (json.dumps(settings, sort_keys=True), is_chat)

            # Group by settings and is_chat flag (only batch compatible requests)
            groups = {}
            for req in batch:
                key = settings_key(req.settings, req.is_chat)
                if key not in groups:
                    groups[key] = []
                groups[key].append(req)

            # Process each group
            for (settings_json, is_chat), reqs in groups.items():
                settings = json.loads(settings_json)

                # Collect all inputs
                all_inputs = []
                request_boundaries = []  # Track (start_idx, end_idx) for each request

                for req in reqs:
                    start_idx = len(all_inputs)

                    if is_chat:
                        # Chat mode: single input per request
                        all_inputs.append(self.apply_chat_template(req.inputs))
                        end_idx = len(all_inputs)
                    elif isinstance(req.inputs, str):
                        # Single string input
                        all_inputs.append(req.inputs)
                        end_idx = len(all_inputs)
                    elif isinstance(req.inputs, list):
                        # Multiple inputs in a single request
                        all_inputs.extend(req.inputs)
                        end_idx = len(all_inputs)
                    else:
                        # Fallback: treat as single input
                        all_inputs.append(req.inputs)
                        end_idx = len(all_inputs)

                    request_boundaries.append((start_idx, end_idx))

                # Run batched inference in thread pool to avoid blocking event loop
                scores, _, preds = await asyncio.get_event_loop().run_in_executor(
                    self.engine._thread_pool,  # ← warmed thread
                    lambda s=settings: self.engine.infer_list(all_inputs, settings=s),
                )

                # Distribute results back to individual requests using boundaries
                for req, (start_idx, end_idx) in zip(reqs, request_boundaries):
                    req_scores = scores[start_idx:end_idx]
                    req_preds = preds[start_idx:end_idx]
                    req.future.set_result((req_scores, req_preds))

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all requests in batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def batch_processor(self):
        """
        Continuously collect and process requests in batches.
        """
        while True:
            batch = []
            deadline = None

            # Get first request (blocking)
            try:
                req = await self.request_queue.get()
                batch.append(req)
                deadline = time.time() + self.batch_timeout
            except Exception:
                continue

            # Collect more requests until timeout or batch full
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    timeout = max(0, deadline - time.time())
                    req = await asyncio.wait_for(self.request_queue.get(), timeout=timeout)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Process batch
            await self._process_batch(batch)

    def maybe_retrieve_model(self):
        """
        Download the model if it's not available locally.
        """
        from huggingface_hub import HfApi, snapshot_download

        hf_api = HfApi()
        try:
            hf_api.model_info(self.model_path)
        except Exception:
            self.local_path = os.path.expandvars(self.model_path)
        else:
            self.local_path = os.path.expandvars(os.path.join(self.models_root, self.model_path))
            logger.info(f"Downloading {self.model_path} from huggingface, " f"to local directory {self.local_path}")
            snapshot_download(repo_id=self.model_path, local_dir=self.local_path)

    async def ensure_batch_processor(self):
        """
        Start batch processor if not already running.
        Must be called from async context.
        """
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self.batch_processor())
            logger.info(f"Started batch processor for model {self.model_id}")

    def load(self):
        """
        Create the inference engine.
        """
        self.maybe_retrieve_model()
        self.get_config()
        self.engine = InferenceEnginePY(self.config)
        self.loaded = True
        logger.info(f"Loaded model {self.model_id} from: {self.model_path}")

    def unload(self):
        """
        Not super clean, we might want to do better some day...
        """
        # Cancel batch processor if running
        if self.processing_task is not None and not self.processing_task.done():
            self.processing_task.cancel()
            self.processing_task = None

        # Clear any pending requests
        while not self.request_queue.empty():
            try:
                req = self.request_queue.get_nowait()
                if not req.future.done():
                    req.future.set_exception(Exception("Model unloaded"))
            except Exception:
                break
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        self.engine = None
        self.loaded = False
        logger.info(f"Unloaded model {self.model_id}")

    def get_model_limits(self):
        """
        Return ``(max_tokens, max_input_tokens)`` for this model.

        *max_tokens* is the maximum number of generation tokens (``max_length``
        from the inference config).

        *max_input_tokens* is the maximum number of tokens that fit in the
        context window before generation.  It is derived from the effective
        context length:

        1. If ``context_length`` is set explicitly in the inference config, use
           that value.
        2. Otherwise fall back to ``original_max_position_embeddings`` from the
           model's RoPE configuration.
        3. If neither is available, default to 0 (unknown).

        Returns:
            tuple[int, int]: ``(max_tokens, max_input_tokens)``
        """
        if not self.loaded or self.engine is None:
            return 0, 0
        predictor = getattr(self.engine, "predictor", None)
        max_tokens = int(getattr(self.config, "max_length", 0) or 0)
        context_length = int(getattr(self.config, "context_length", 0) or 0)
        if context_length <= 0:
            # Fall back to original_max_position_embeddings from rope config
            rope = getattr(getattr(predictor, "model", None), "decoder", None)
            rope = getattr(rope, "rope", None)
            rope_cfg = getattr(getattr(rope, "model_config", None), "rope_config", None)
            context_length = int(getattr(rope_cfg, "original_max_position_embeddings", 0) or 0)
        max_input_tokens = max(0, context_length - max_tokens)
        return max_tokens, max_input_tokens

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in *text* using the model's tokenizer.

        The method walks the engine's transform pipeline to find the first
        ``TokenizerTransform`` and uses it to tokenize the string.  If no
        tokenizer is found (e.g. the model uses whitespace splitting), it falls
        back to ``estimate_tokens`` (≈4 chars per token).

        Args:
            text (str): The text to tokenize.

        Returns:
            int: Token count.
        """
        from eole.transforms.tokenize import TokenizerTransform

        if not self.loaded or self.engine is None:
            return estimate_tokens(text)
        transform_pipe = getattr(self.engine, "transform_pipe", None)
        if transform_pipe is not None:
            for transform in getattr(transform_pipe, "transforms", []):
                if isinstance(transform, TokenizerTransform):
                    try:
                        return len(transform._tokenize(text, side="src"))
                    except Exception:
                        break
        return estimate_tokens(text)

    def apply_chat_template(self, inputs, tools=None, tool_choice=None, enable_thinking=False, reasoning_effort=None):
        """
        Render the model input based on the model chat template
        and the request inputs.

        Optional *tools* (list of OpenAI-style function tool dicts) and
        *tool_choice* are forwarded as Jinja2 template variables so that
        models whose templates understand them can emit the appropriate
        prompting tokens.

        HuggingFace template compatibility:
        - ``strftime_now`` global is available for templates that embed the
          current date/time (e.g. Llama 3.3).
        - ``{% generation %}`` blocks are stripped before rendering; they mark
          where generation starts in HF's ``apply_chat_template`` but are not
          standard Jinja2 and are not needed for inference.
        - ``tojson`` filter is provided for templates that serialise objects.
        """

        def raise_exception(message):
            raise TemplateError(message)

        chat_template = self.config.chat_template
        if chat_template is None:
            # Fall back to a standalone chat_template.jinja file in the model
            # directory (used by some modern HF models that don't embed the
            # template in tokenizer_config.json or config.json).
            jinja_file = os.path.join(self.local_path, "chat_template.jinja")
            if os.path.exists(jinja_file):
                with open(jinja_file, encoding="utf-8") as f:
                    chat_template = f.read()
            else:
                raise TemplateError(
                    f"Model '{self.model_id}' has no chat_template configured. "
                    "Set chat_template in the model's inference config or provide "
                    "a chat_template.jinja file in the model directory."
                )
        # Modern HuggingFace models store chat_template as a list of named
        # templates, e.g. [{"name": "default", "template": "..."}, ...].
        # Extract the "default" entry, or fall back to the first entry.
        if isinstance(chat_template, list):
            # Use .get() to safely handle list items that may lack a "template" key.
            template_str = next(
                (t.get("template") for t in chat_template if isinstance(t, dict) and t.get("name") == "default"),
                None,
            )
            if template_str is None and chat_template:
                first = chat_template[0]
                template_str = (
                    first.get("template") if isinstance(first, dict) else (first if isinstance(first, str) else None)
                )
            if not isinstance(template_str, str):
                raise TemplateError(f"Model '{self.model_id}': chat_template list contains no usable template string.")
            chat_template = template_str

        # Guard against any remaining non-string value (e.g. dict or bytes from
        # unexpected config formats) to give a clear error instead of a cryptic
        # "Can't compile non template nodes" TypeError from Jinja2.
        if not isinstance(chat_template, str):
            raise TemplateError(
                f"Model '{self.model_id}': chat_template has unexpected type "
                f"'{type(chat_template).__name__}' — expected a string. "
                "Check the model's config.json or chat_template.jinja file."
            )

        inputs = _normalize_developer_role(inputs, chat_template)

        # Strip HuggingFace {% generation %} markers — they are used by the
        # HF tokenizer library to mark the generation boundary but are not
        # valid Jinja2 tags and are not needed for inference rendering.
        chat_template = chat_template.replace("{% generation %}", "")

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        # HF templates often call strftime_now() to embed the current date.
        jinja_env.globals["strftime_now"] = datetime.datetime.now().strftime
        # tojson is a common filter used in HF templates to serialise objects.
        jinja_env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)

        template = jinja_env.from_string(chat_template)
        render_kwargs: dict = {
            "messages": inputs,
            "bos_token": "",  # handled in numericalize
            "eos_token": "",  # handled by stop conditions; use literal tokens in template
            "add_generation_prompt": True,
            # Many Qwen3 / "thinking" model templates gate on this value.
            "enable_thinking": enable_thinking,
        }
        if reasoning_effort is not None:
            render_kwargs["reasoning_effort"] = reasoning_effort
        if tools is not None:
            render_kwargs["tools"] = tools
        if tool_choice is not None:
            render_kwargs["tool_choice"] = tool_choice
        rendered_output = template.render(**render_kwargs)

        # _log_json_payload(f"RENDERED PROMPT [{self.model_id}]",rendered_output,)

        return rendered_output

    async def infer_async(self, inputs, settings={}, is_chat=False):
        """
        Queue inference request and wait for result.
        """
        # Ensure model is loaded (sync operation)
        if not self.loaded:
            self.load()

        # Ensure batch processor is running (async operation)
        await self.ensure_batch_processor()

        future = asyncio.Future()
        req = QueuedRequest(inputs=inputs, settings=settings, is_chat=is_chat, future=future, timestamp=time.time())
        await self.request_queue.put(req)
        return await future
