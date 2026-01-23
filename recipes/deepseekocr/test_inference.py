# flake8: noqa
import os
from rich import print


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(
        model_path=os.path.join(mydir, "DeepSeek-OCR"),
        src="dummy",
        self_attn_backend="flash",
        max_length=8192,
        world_size=1,
        gpu_ranks=[0],
        compute_dtype="bf16",
        top_p=0.0,
        top_k=1,
        temperature=0.0,
        beam_size=1,
        seed=12,
        batch_size=4,
        batch_type="sents",
        report_time=True,
        fuse_gate=True,
        fuse_kvq=True,
    )

    config.data_type = "image"
    return config


def build_test_inputs():
    return [
        {
            "text": "{image}\nFree OCR.",
            "images": {"image": "eole/tests/data/images/deepseekpaper.png"},
        },
        {
            "text": "{image}\n<|grounding|>Convert the document to markdown.",
            "images": {"image": "eole/tests/data/images/deepseekpaper.png"},
        },
        {
            "text": "{image}\nFree OCR.",
            "images": {"image": "eole/tests/data/images/deepseekpapertable.png"},
        },
        {
            "text": "{image}\n<|grounding|>Convert the document to markdown.",
            "images": {"image": "eole/tests/data/images/deepseekpapertable.png"},
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
        print("######################## first pass - full warm up #####################")
        _, _, pred = engine.infer_list(test_input)
        print("######################## first pass - actual run #######################")
        _, _, pred = engine.infer_list(test_input)

        postprocess_and_print(pred, test_input)

    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
