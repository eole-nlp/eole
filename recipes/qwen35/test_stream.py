# flake8: noqa
"""
Streaming inference example for Qwen 3.5.

Similar to test_inference2.py, but uses engine.infer_list_stream() so that
tokens are printed to the terminal as soon as they are generated instead of
waiting for the full response.

Usage::

    EOLE_MODEL_DIR=/path/to/models python recipes/qwen35/test_stream.py
"""
import os
import sys


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(
        model_path=os.path.join(mydir, "qwen3.5-4B"),
        src="dummy",
        self_attn_backend="flash",
        max_length=4096,
        gpu_ranks=[0],
        compute_dtype="bf16",
        top_k=0,
        top_p=0.0,
        temperature=1.0,
        beam_size=1,
        seed=42,
        batch_size=1,
        batch_type="sents",
        report_time=True,
        fuse_kvq=False,
    )

    return config


def build_test_inputs():
    return [
        "<|im_start|>user\nGenerate a 200 word text talking about George Orwell.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ]


def main():
    from eole.inference_engine import InferenceEnginePY

    config = build_config()
    engine = InferenceEnginePY(config)

    try:
        test_inputs = build_test_inputs()

        for i, src in enumerate(test_inputs):
            print(f'{"#" * 40} example {i} {"#" * 40}')
            # Stream tokens one by one, printing each chunk as it arrives.
            for chunk in engine.infer_list_stream(src):
                chunk = chunk.replace("｟newline｠", "\n")
                print(chunk, end="", flush=True)
            # Newline after the streamed output
            print()

    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
