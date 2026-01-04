# flake8: noqa
import os
from rich import print
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY

mydir = os.getenv("EOLE_MODEL_DIR")

config = PredictConfig(
    model_path=os.path.join(mydir, "gemma3-27b-it"),
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
    top_p=0.8,
    temperature=0.35,
    beam_size=1,
    seed=42,
    batch_size=1,
    batch_type="sents",
    report_time=True,
    fuse_kvq=True,
)

config.data_type = "image"
engine = InferenceEnginePY(config)

test_input = [
    {
        "text": "<start_of_turn>user\nList the top 5 countries in Europe with the highest GDP from this image\n{image1}<end_of_turn><start_of_turn>model\n",
        "images": {"image1": "eole/tests/data/images/gdp.png"},
    },
    {
        "text": "<start_of_turn>user\nWhen did things start to go wrong for dark dragon?\n{image1}<end_of_turn><start_of_turn>model\n",
        "images": {"image1": "eole/tests/data/images/loss_curve.jpg"},
    },
    # {
    #     "text": "<start_of_turn>user\nIs this person really big, or is this building just super small?\n{image1}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/pisa_2.jpg"
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nCombine information in both the tables into a single markdown table\n{image1}\n{image2}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/table1.png",
    #         "image2": "../../eole/tests/data/images/table2.png",
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nCombine information in both the tables into a single markdown table\n{image1}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/multi-images.png"
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nDescribe the images.\n{image1}\n{image2}\n{image3}\n{image4}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/image1.png",
    #         "image2": "../../eole/tests/data/images/image2.png",
    #         "image3": "../../eole/tests/data/images/image3.png",
    #         "image4": "../../eole/tests/data/images/image4.png",
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nCombine information in both the tables into a single markdown table\n{image1}{image2}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/table1.png",
    #         "image2": "../../eole/tests/data/images/table2.png"
    #     }
    # },
]

_, _, preds = engine.infer_list(test_input)

for i in range(len(test_input)):
    print(f'{"#" * 40} example {i} {"#" * 40}')
    print(preds[i][0].replace("｟newline｠", "\n"))
