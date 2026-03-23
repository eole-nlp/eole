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
import time
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
    repetition_penalty: float = 1.0,
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
        repetition_penalty: Repetition penalty factor (1.0 = no penalty).

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
        "repetition_penalty": repetition_penalty,
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


def _format_stats(
    prompt_tokens: int,
    completion_tokens: int,
    elapsed_s: float,
) -> str:
    """Return a human-readable stats string."""
    # Guard against division by zero or unrealistically tiny elapsed times
    tok_sec = completion_tokens / elapsed_s if elapsed_s > 0.01 else 0.0
    lines = [
        f"⏱  Generation time : {elapsed_s:.2f} s",
        f"🚀 Throughput       : {tok_sec:.1f} tok/s",
        f"📥 Prompt tokens    : {prompt_tokens}",
        f"📤 Completion tokens: {completion_tokens}",
        f"📊 Total tokens     : {prompt_tokens + completion_tokens}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------


def build_app(default_api_url: str, default_model: str) -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    # Fetch initial model list once so both the dropdown value and choices are set.
    initial_models = get_models(default_api_url)
    initial_model_value = default_model if default_model else (initial_models[0] if initial_models else None)

    with gr.Blocks(title="Eole Chatbot") as app:
        gr.Markdown("# Eole Chatbot")

        # ── Top bar: server URL + model picker ──────────────────────────────
        with gr.Row():
            api_url_input = gr.Textbox(
                label="Server URL",
                value=default_api_url,
                scale=3,
                interactive=True,
            )
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=initial_models,
                value=initial_model_value,
                scale=2,
                interactive=True,
            )
            refresh_btn = gr.Button("↻ Refresh models", scale=1)

        # ── Main content: chat on the left, panels on the right ──────────────
        with gr.Row():
            # ── Left column: chat widget + input ────────────────────────────
            with gr.Column(scale=3):
                chatbot_widget = gr.Chatbot(
                    elem_id="chatbot",
                    latex_delimiters=LATEX_DELIMITERS,
                    height=600,
                )
                chatbot_widget.like(vote, None, None)

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message and press Enter…",
                        label="Message",
                        show_label=False,
                        scale=9,
                        lines=1,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

                clear_btn = gr.Button("🗑 Clear conversation")

            # ── Right column: settings + stats ───────────────────────────────
            with gr.Column(scale=1, min_width=220):
                with gr.Accordion("⚙ Settings", open=True):
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.05,
                        label="Temperature",
                    )
                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        label="Top-p",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=20,
                        step=1,
                        label="Top-k",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=64,
                        maximum=8192,
                        value=4096,
                        step=64,
                        label="Max tokens",
                    )
                    repetition_penalty_slider = gr.Slider(
                        minimum=0.9,
                        maximum=2.0,
                        value=1.0,
                        step=0.01,
                        label="Repetition penalty",
                    )

                with gr.Accordion("📊 Stats", open=True):
                    stats_box = gr.Textbox(
                        label="Inference stats",
                        value="",
                        lines=6,
                        interactive=False,
                    )

        # ── History state ────────────────────────────────────────────────────
        history_state = gr.State([])

        # ── Event handlers ───────────────────────────────────────────────────

        def user_submit(user_message: str, history: list[dict]) -> tuple[str, list[dict], list[dict]]:
            """Append user message to history; clear the input box."""
            if not user_message.strip():
                return user_message, history, history
            history = history + [{"role": "user", "content": user_message}]
            return "", history, history

        def bot_stream(
            history: list[dict],
            api_url: str,
            model: str,
            temperature: float,
            top_p: float,
            top_k: int,
            max_tokens: int,
            repetition_penalty: float,
        ) -> Generator[tuple[list[dict], str], None, None]:
            """Stream the assistant reply and update the stats panel."""
            messages = [{"role": m["role"], "content": m["content"]} for m in history]

            history = history + [{"role": "assistant", "content": ""}]
            accumulated = ""
            token_count = 0
            start = time.time()
            first_token_time = None
            try:
                for chunk in stream_chat_completions(
                    api_url=api_url,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=int(top_k),
                    max_tokens=int(max_tokens),
                    repetition_penalty=repetition_penalty,
                ):
                    accumulated += chunk
                    # Each SSE chunk corresponds to approximately one token in
                    # typical LLM streaming; counting chunks is a better proxy
                    # than splitting by whitespace.
                    if first_token_time is None:
                        first_token_time = time.time()  # start of decode phase
                    token_count += 1
                    history[-1]["content"] = accumulated
                    yield history, ""
            except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
                error_msg = f"[Error communicating with server: {exc}]"
                history[-1]["content"] = error_msg
                yield history, ""
                return

            elapsed = time.time() - (first_token_time or start)
            # Estimate prompt token count: ~4 characters per token is a common
            # rule-of-thumb approximation across most BPE-based tokenizers.
            prompt_text = " ".join(m["content"] for m in messages)
            prompt_tokens = max(1, len(prompt_text) // 4)
            stats = _format_stats(
                prompt_tokens=prompt_tokens,
                completion_tokens=token_count,
                elapsed_s=elapsed,
            )
            yield history, stats

        # Wire up send button and Enter key
        submit_inputs = [
            history_state,
            api_url_input,
            model_dropdown,
            temperature_slider,
            top_p_slider,
            top_k_slider,
            max_tokens_slider,
            repetition_penalty_slider,
        ]

        (
            msg_input.submit(
                fn=user_submit,
                inputs=[msg_input, history_state],
                outputs=[msg_input, history_state, chatbot_widget],
                queue=False,
            ).then(
                fn=bot_stream,
                inputs=submit_inputs,
                outputs=[chatbot_widget, stats_box],
            )
        )

        (
            send_btn.click(
                fn=user_submit,
                inputs=[msg_input, history_state],
                outputs=[msg_input, history_state, chatbot_widget],
                queue=False,
            ).then(
                fn=bot_stream,
                inputs=submit_inputs,
                outputs=[chatbot_widget, stats_box],
            )
        )

        def clear_history() -> tuple[list, list, str]:
            return [], [], ""

        clear_btn.click(fn=clear_history, outputs=[history_state, chatbot_widget, stats_box])

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
