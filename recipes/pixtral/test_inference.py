# flake8: noqa
import os
from rich import print


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(
        model_path=os.path.join(mydir, "mistralai/pixtral-12b"),
        src="dummy",
        self_attn_backend="flash",
        max_length=2048,
        gpu_ranks=[0],
        compute_dtype="bf16",
        top_p=0.8,
        temperature=0.35,
        beam_size=1,
        seed=42,
        batch_size=2,
        batch_type="sents",
        report_time=True,
        fuse_kvq=True,
        fuse_gate=True,
    )

    config.data_type = "image"
    return config


def build_test_inputs():
    return [
        {
            "text": "[INST]List the top 5 countries in Europe with the highest GDP\n{image1}[/INST]",
            "images": {"image1": "eole/tests/data/images/gdp.png"},
        },
        {
            "text": "[INST]When did things start to go wrong for dark dragon?\n{image1}[/INST]",
            "images": {"image1": "eole/tests/data/images/loss_curve.jpg"},
        },
        # {
        #     "text": "<s>[INST]Is this person really big, or is this building just super small?\n{image1}[/INST]",
        #     "images": {
        #         "image1": "../../eole/tests/data/images/pisa_2.jpg"
        #     }
        # },
        # {
        #     "text": "<s>[INST]Combine information in both the tables into a single markdown table\n{image1}\n{image2}[/INST]",
        #     "images": {
        #         "image1": "../../eole/tests/data/images/table1.png",
        #         "image2": "../../eole/tests/data/images/table2.png"
        #     }
        # },
        # {
        #     "text": "<s>[INST]Combine information in both the tables into a single markdown table\n{image1}[/INST]",
        #     "images": {
        #         "image1": "../../eole/tests/data/images/multi-images.png"
        #     }
        # },
        # {
        #     "text": "<s>[INST]Describe the images.\n{image1}\n{image2}\n{image3}\n{image4}[/INST]",
        #     "images": {
        #         "image1": "../../eole/tests/data/images/image1.png",
        #         "image2": "../../eole/tests/data/images/image2.png",
        #         "image3": "../../eole/tests/data/images/image3.png",
        #         "image4": "../../eole/tests/data/images/image4.png",
        #     }
        # },
        # {
        #     "text": "<s>[INST]Combine information in both the tables into a single markdown table\n{image1}{image2}[/INST]",
        #     "images": {
        #         "image1": "../../eole/tests/data/images/table1.png",
        #         "image2": "../../eole/tests/data/images/table2.png"
        #     }
        # },
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
