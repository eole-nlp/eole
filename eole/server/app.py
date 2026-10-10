"""FastAPI application construction and native Eole routes."""

from typing import List, Union

from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

import eole
from eole.config.inference import DecodingConfig
from eole.constants import DefaultTokens
from eole.server.model import Server
from eole.server.openai_chat import register_openai_chat
from eole.server.anthropic import register_anthropic
from eole.server.responses import register_responses

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
    async def infer(
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
            inputs = request.messages
        model_id = request.model
        # automatically grab anything that is not model/inputs
        # (we could probably rely on pydantic model once properly implemented)
        non_settings_keys = ["inputs", "messages", "model"]
        settings = {k: v for k, v in request.model_dump().items() if k not in non_settings_keys}

        await server.maybe_load_model(model_id)
        scores, preds = await server.models[model_id].infer_async(
            inputs,
            settings=settings,
            is_chat=isinstance(request, ChatRequest),
        )
        response = {"predictions": preds, "scores": scores}
        return response

    register_openai_chat(app, server)
    register_anthropic(app, server)
    register_responses(app, server)
    return app
