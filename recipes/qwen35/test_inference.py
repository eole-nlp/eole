# flake8: noqa
import os
from rich import print


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(
        model_path=os.path.join(mydir, "qwen3.5-27B"),
        src="dummy",
        self_attn_backend="flash",
        max_length=2048,
        gpu_ranks=[0],
        quant_type="bnb_NF4",  # HF default, using it for initial reproducibility checks
        quant_layers=[
            "gate_up_proj",
            "down_proj",
            "up_proj",
        ],
        compute_dtype="bf16",
        top_k=20,
        top_p=0.95,
        temperature=1.0,
        beam_size=1,
        seed=42,
        batch_size=1,
        batch_type="sents",
        report_time=True,
        fuse_kvq=False,
    )

    config.data_type = "image"
    return config


def build_test_inputs():
    return [
        {
            "text": "<|im_start|>user\nList the top 5 countries in Europe with the highest GDP from this image. Just output 5 lines.\n{image1}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "images": {"image1": "eole/tests/data/images/gdp.png"},
        },
        {
            "text": "<|im_start|>user\nWhen did things start to go wrong for dark dragon?\n{image1}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "images": {"image1": "eole/tests/data/images/loss_curve.jpg"},
        },
    ]


def postprocess_and_print(pred, test_input):

    for i in range(len(test_input)):
        print(f'{"#" * 40} example {i} {"#" * 40}')
        text = pred[i][0]
        print(text.replace("｟newline｠", "\n"))


def main():
    from eole.inference_engine import InferenceEnginePY

    config = build_config()
    engine = InferenceEnginePY(config)

    try:
        test_input = build_test_inputs()
        _, _, pred = engine.infer_list(test_input)

        postprocess_and_print(pred, test_input)

    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
