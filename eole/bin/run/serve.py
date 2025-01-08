#!/usr/bin/env python

import os
import time
import gc
import yaml

from typing import List, Union

import torch
import uvicorn

from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

import eole
from eole.inference_engine import InferenceEnginePY
from eole.config.run import PredictConfig
from eole.config.inference import DecodingConfig
from eole.bin import register_bin, BaseBin
from eole.utils.logging import logger
from eole.constants import DefaultTokens

STATUS_OK = "ok"
STATUS_ERROR = "error"


class TextRequest(DecodingConfig):
    """
    Standard text "completion" request
    (as well as encoder/decoder models e.g. translation).
    """

    model: int | str = Field(description="Model identifier from server configuration.")
    inputs: Union[str, List[str]] = Field(
        description="List of inputs to run inference on. "
        "A single string will be automatically cast to a single item list."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama3-8b-instruct",
                "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a funny guy.<|eot_id|><|start_header_id|>user<|end_header_id|>Tell me a joke :)<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # noqa: E501
            }
        }


class TextResponse(BaseModel):
    """
    Response of TextRequest.
    """

    predictions: List[List[str]] = Field(description="List of prediction(s) for each input(s).")
    scores: List[List[float]] = Field(description="Pred scores from the model for each prediction.")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    [
                        "\n\nHere's one:\n\nWhy couldn't the bicycle stand up by itself?\n\n(wait for it...)\n\nBecause it was two-tired!\n\nHope that made you laugh!"  # noqa: E501
                    ]
                ],
                "scores": [[-0.040771484375]],
            }
        }

    @model_validator(mode="after")
    def _validate_response(self):
        """
        Automatically apply some formatting to the provided text response.
        This logic might be moved elsewhere at some point.
        """
        self.predictions = [[pred.replace(DefaultTokens.SEP, "\n") for pred in preds] for preds in self.predictions]
        return self


class ChatRequest(DecodingConfig):
    """
    Request format for chat-based interactions.
    """

    model: int | str = Field(description="Model identifier from server configuration.")
    messages: List[dict] = Field(description="List of message dictionaries with 'role' and 'content' keys.")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "llama3-8b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a funny guy."},
                    {"role": "user", "content": "Tell me a joke :)"},
                ],
            }
        }


# TODO: specific response model for chat mode?
# class ChatResponse(BaseModel):
#     choices: List[dict]


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

    def maybe_load_model(self, model_id_to_load):
        """
        Very naive method to ensure a single model is loaded for now.
        """
        for model_id, model in self.models.items():
            if model_id != model_id_to_load:
                model.unload()


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
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        self.engine = None
        self.loaded = False
        logger.info(f"Unloaded model {self.model_id}")

    def apply_chat_template(self, inputs):
        """
        Render the model input based on the model chat template
        and the request inputs.
        """

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        template = jinja_env.from_string(self.config.chat_template)
        rendered_output = template.render(
            **{
                "messages": inputs,
                "bos_token": "",  # handled in numericalize
                "add_generation_prompt": True,
            }
        )
        return rendered_output

    def infer(self, inputs, settings={}, is_chat=False):
        """
        Run inference on the given inputs.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if not (self.loaded):
            self.load()
        if is_chat:
            assert hasattr(self.config, "chat_template"), "Chat requests can't be performed without a chat_template."
            inputs = [self.apply_chat_template(inputs)]
        scores, _, preds = self.engine.infer_list(inputs, settings=settings)
        return scores, preds


def create_app(config_file):
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title="Eole Inference Server",
        version=eole.__version__,
        summary="A simple inference server to expose various models.",
        description="",  # TODO
    )

    server = Server()
    server.start(config_file)

    @app.get("/")
    def root(request: Request):
        """
        Root endpoint returning HTML content to help users find the docs.
        """
        html_content = f"""
        <html>
            <head>
                <title>Eole Server</title>
            </head>
            <body>
                <h1>Eole Server</h1>
                <p>Probably not what you're looking for.</p>
                <p>API docs --> <a href="{request.url}docs">{request.url}docs</a>.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

    @app.get("/models")
    def models():
        """
        Return available models currently exposed.
        """
        models = server.available_models()
        out = {"models": models}
        return out

    @app.post("/unload_model")
    def unload_model(model_id):
        """
        Unload a specific model.
        """
        server.models[model_id].unload()

    @app.get("/health")
    def health():
        """
        Health check endpoint.
        """
        out = {}
        out["status"] = STATUS_OK
        return out

    @app.post("/infer", response_model=TextResponse)
    def infer(
        request: Union[TextRequest, ChatRequest] = Body(
            openapi_examples={
                "text_request": {
                    "summary": "Text Request Example",
                    "description": "A sample text request",
                    "value": TextRequest.Config.json_schema_extra["example"],
                },
                "chat_request": {
                    "summary": "Chat Request Example",
                    "description": "A sample chat request",
                    "value": ChatRequest.Config.json_schema_extra["example"],
                },
            },
        ),
    ):
        """
        Run inference on the given input.
        """
        if isinstance(request, TextRequest):
            inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
        else:  # ChatRequest
            # no batch support right now
            inputs = request.messages
        model_id = request.model
        # automatically grab anything that is not model/inputs
        # (we could probably rely on pydantic model once properly implemented)
        non_settings_keys = ["inputs", "messages", "model"]
        settings = {k: v for k, v in request.model_dump().items() if k not in non_settings_keys}
        # TODO: move this in some `infer` method in the `Server` class?
        server.maybe_load_model(model_id)
        scores, preds = server.models[model_id].infer(
            inputs,
            settings=settings,
            is_chat=isinstance(request, ChatRequest),
        )
        # returned scores are tensors which we need to cast (not anymore?)
        # scores = [[score.item() for score in score_list] for score_list in scores]
        response = {"predictions": preds, "scores": scores}
        return response

    # @app.post("/openai/chat/completions")
    # def openai_chat(request: ChatRequest):
    #     """
    #     Simulate an OpenAI Request.
    #     The idea is to make this a viable alternative as a drop-in
    #     replacement for OpenAI or other LLM stacks.
    #     """
    #     pass

    return app


@register_bin(name="serve")
class Serve(BaseBin):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--config",
            "-config",
            "-c",
            default="./server_conf.yaml",
            help="Path of server YAML config file.",
        )
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default="5000")

    @classmethod
    def run(cls, args):
        app = create_app(args.config)
        uvicorn.run(app=app, host=args.host, port=args.port, log_level="info")
