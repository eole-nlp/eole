#!/usr/bin/env python

import os
import time
import gc
import yaml

from typing import List, Union

import torch
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from eole.inference_engine import InferenceEnginePY
from eole.config.run import PredictConfig
from eole.config.inference import DecodingConfig
from eole.bin import register_bin, BaseBin
from eole.utils.logging import logger

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


class TextResponse(BaseModel):
    """
    Response of TextRequest.
    """

    predictions: List[List[str]] = Field(
        description="List of prediction(s) for each input(s)."
    )
    scores: List[List[float]] = Field(
        description="Pred scores from the model for each prediction."
    )


# class ChatRequest(DecodingConfig):
#     model: str
#     messages: List[dict]


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
        with open(server_config_path) as f:
            server_config = yaml.safe_load(f)
        self.models_root = server_config["models_root"]
        for model in server_config["models"]:
            # instantiate models
            # add some safeguards here, download from HF, etc.
            model_id = model["id"]
            model_path = model["path"]
            self.models[model_id] = Model(
                model_path=model_path,
                models_root=self.models_root,
                model_type=model.get("model_type", "default"),
            )
            if model.get("preload", False):
                self.models[model_id].start_engine()

    def available_models(self):
        models = []
        for model_id, model in self.models.items():
            models.append({"id": model_id})
        return models


class Model(object):
    def __init__(
        self, model_path=None, preload=False, models_root=None, model_type=False
    ):
        self.loaded = False
        self.engine = None
        self.preload = preload
        self.models_root = models_root
        self.model_path = model_path
        self.local_path = None
        self.model_type = model_type

    def get_config(self):
        # transforms and inference settings are retrieved from the model config for now
        self.config = PredictConfig(
            src="dummy",
            model_path=self.local_path,
            # TODO improve this
            gpu_ranks=[0],
            world_size=1,
        )

    def override_opts(self):
        """
        Potentially override some opts from a config file?
        """
        pass

    def maybe_retrieve_model(self):
        from huggingface_hub import HfApi, snapshot_download

        hf_api = HfApi()
        try:
            hf_api.model_info(self.model_path)
        except Exception:
            self.local_path = os.path.expandvars(self.model_path)
        else:
            self.local_path = os.path.expandvars(
                os.path.join(self.models_root, self.model_path)
            )
            logger.info(
                f"Downloading {self.model_path} from huggingface, "
                f"to local directory {self.local_path}"
            )
            snapshot_download(repo_id=self.model_path, local_dir=self.local_path)

    def start_engine(self):
        """
        We might want to call this "load"...
        """

        self.maybe_retrieve_model()
        self.get_config()
        self.engine = InferenceEnginePY(self.config)
        self.loaded = True

    def unload(self):
        """
        stop_engine if start_engine is not renamed...
        Not super clean, we might want to do better some day...
        """
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        self.engine = None
        self.loaded = False

    def infer(self, inputs, settings={}):
        if type(inputs) == str:
            inputs = [inputs]
        if not (self.loaded):
            self.start_engine()
        scores, _, preds = self.engine.infer_list(inputs, settings=settings)
        return scores, preds


def create_app(config_file):
    app = FastAPI()

    server = Server()
    server.start(config_file)

    @app.get("/")
    def root(request: Request):
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
        server.models[model_id].unload()

    @app.get("/health")
    def health():
        out = {}
        out["status"] = STATUS_OK
        return out

    @app.post("/infer")
    def infer(request: TextRequest, response_model=TextResponse):
        if isinstance(request.inputs, str):
            request.inputs = [request.inputs]
        model_id = request.model
        inputs = request.inputs
        # automatically grab anything that is not model/inputs
        # (we could probably rely on pydantic model once properly implemented)
        non_settings_keys = ["inputs", "model"]
        settings = {
            k: v for k, v in request.model_dump().items() if k not in non_settings_keys
        }
        scores, preds = server.models[model_id].infer(inputs, settings=settings)
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
    #     The actual chat -> prompt conversion might depend on the model,
    # and could be defined in the inference.json config for instance.
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
