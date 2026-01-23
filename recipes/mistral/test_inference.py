# flake8: noqa
import os
from rich import print

sys_prmt = "[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER\n\nFirst draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response.[/SYSTEM_PROMPT]"


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(  # flake8: noqa
        model_path=os.path.join(mydir, "mistralai/Ministral-3-14B-Reasoning-2512"),
        src="dummy",
        max_length=4096,
        gpu_ranks=[0],
        # quant_type="bnb_NF4",  # HF default, using it for initial reproducibility checks
        quant_layers=[
            "gate_up_proj",
            "down_proj",
            "up_proj",
            "linear_values",
            "linear_query",
            "linear_keys",
            "final_linear",
            "w_in",
            "w_out",
        ],
        compute_dtype="bf16",
        top_p=0.8,
        temperature=0.35,
        beam_size=1,
        seed=42,
        batch_size=2,
        batch_type="sents",
        report_time=True,
        # fuse_kvq=True,
        # fuse_gate=True,
    )

    config.data_type = "image"
    return config


def build_test_inputs():
    return [
        {
            "text": sys_prmt + "[INST]Convert the image to markdown. Tables should be in html.\n{image1}[/INST]",
            "images": {"image1": "eole/tests/data/images/deepseekpaper.png"},
        },
        {
            "text": sys_prmt + "[INST]Convert the image to markdown. Tables should be in html.\n{image1}[/INST]",
            "images": {"image1": "eole/tests/data/images/deepseekpapertable.png"},
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
        text = pred[i][0].replace("[THINK]", "\[THINK]\n\n").replace("[/THINK]", "\[/THINK]\n\n")
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
