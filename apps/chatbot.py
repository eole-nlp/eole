"""Gradio chatbot backed by an eole server with token-by-token streaming.

The app connects to the eole server's OpenAI-compatible
``/v1/chat/completions`` endpoint using ``stream=True`` so that tokens
are rendered in the chat window as soon as they are produced by the model.

Usage::

    python apps/chatbot.py [--api-url http://localhost:5000] [--model qwen3.5-4B] [--share]

The server must be started separately, e.g.::

    eole serve --config server_config.yaml
"""

import argparse
import json
from typing import Generator

import gradio as gr
import requests


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": r"\(", "right": r"\)", "display": False},
    {"left": r"\[", "right": r"\]", "display": True},
]


def stream_chat_completions(
    api_url: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 4096,
) -> Generator[str, None, None]:
    """Yield incremental text chunks from the server's SSE stream.

    Sends a ``/v1/chat/completions`` request with ``stream=True`` and
    parses each ``data:`` line, yielding the ``delta.content`` as it
    arrives.

    Args:
        api_url: Base server URL, e.g. ``http://localhost:5000``.
        model: Model identifier as configured on the server.
        messages: OpenAI-style message list (``[{"role": ..., "content": ...}]``).
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling.
        max_tokens: Maximum number of tokens to generate.

    Yields:
        Incremental text delta strings.
    """
    url = f"{api_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    with requests.post(url, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta_content:
                yield delta_content.replace("｟newline｠", "\n")


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------


def get_models(api_url: str) -> list[str]:
    """Fetch the list of loaded models from the server."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/models", timeout=5)
        response.raise_for_status()
        data = response.json()
        models = [m.get("id", str(m)) for m in data.get("models", [])]
        return models or ["No models found"]
    except (requests.RequestException, ValueError) as exc:
        return [f"Error: {exc}"]


def refresh_models(api_url: str) -> dict:
    models = get_models(api_url)
    return gr.update(choices=models, value=models[0] if models else None)


def vote(data: gr.LikeData) -> None:
    action = "upvoted" if data.liked else "downvoted"
    print(f"User {action}: {data.value!r}")


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------


def build_app(default_api_url: str, default_model: str) -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    def chat_fn(user_message: str, history: list[dict], api_url: str, model: str) -> Generator[str, None, None]:
        """Stream the assistant reply for *user_message* given *history*.

        This function is a generator: Gradio will call ``next()`` on it and
        display each partial string in the chatbot widget, giving the
        appearance of token-by-token streaming.
        """
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        messages.append({"role": "user", "content": user_message})

        accumulated = ""
        try:
            for chunk in stream_chat_completions(api_url=api_url, model=model, messages=messages):
                accumulated += chunk
                yield accumulated
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            yield f"[Error communicating with server: {exc}]"

    with gr.Blocks(title="Eole Chatbot") as app:
        gr.Markdown("# Eole Chatbot")

        with gr.Row():
            api_url_input = gr.Textbox(
                label="Server URL",
                value=default_api_url,
                scale=3,
                interactive=True,
            )
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=get_models(default_api_url),
                value=default_model or None,
                scale=2,
                interactive=True,
            )
            refresh_btn = gr.Button("↻ Refresh models", scale=1)

        chatbot_widget = gr.Chatbot(
            elem_id="chatbot",
            latex_delimiters=LATEX_DELIMITERS,
            height=700,
        )
        chatbot_widget.like(vote, None, None)

        gr.ChatInterface(
            fn=chat_fn,
            chatbot=chatbot_widget,
            additional_inputs=[api_url_input, model_dropdown],
        )

        refresh_btn.click(fn=refresh_models, inputs=[api_url_input], outputs=[model_dropdown])

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eole streaming chatbot (Gradio)")
    parser.add_argument(
        "--api-url",
        default="http://localhost:5000",
        help="Base URL of the eole server (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model identifier (default: first model returned by server)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(default_api_url=args.api_url, default_model=args.model)
    app.launch(share=args.share)
