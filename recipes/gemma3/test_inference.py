# flake8: noqa

from rich import print
from eole.config.run import *
from eole.inference_engine import InferenceEnginePY

config = PredictConfig(
    model_path="./gemma3-27b-it",
    src="dummy",
    max_length=600,
    gpu_ranks=[0],
    # quant_type="bnb_NF4",
    quant_type="bnb_FP4",  # HF default, using it for initial reproducibility checks
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
    batch_size=1,
    batch_type="sents",
)

print(config)

config.data_type = "image"
engine = InferenceEnginePY(config)

print(engine.predictor.model)
engine.predictor.model.count_parameters()

test_input = [
    # {
    #    "text": "<start_of_turn>user\nList the top 5 countries in Europe with the highest GDP from this image\n{image1}<end_of_turn><start_of_turn>model\n",
    #    "images": {"image1": "../../tests/data/gdp.png"},
    # },
    # {
    #     "text": "<start_of_turn>user\nWhen did things start to go wrong for dark dragon?\n{image1}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../tests/data/loss_curve.jpg"
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nIs this person really big, or is this building just super small?\n{image1}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../tests/data/pisa_2.jpg"
    #     }
    # },
    {
        "text": "<start_of_turn>user\nCombine information in both the tables into a single markdown table\n{image1}\n{image2}<end_of_turn><start_of_turn>model\n",
        "images": {"image1": "../../tests/data/table1.png", "image2": "../../tests/data/table2.png"},
    },
    # {
    #     "text": "<start_of_turn>user\nCombine information in both the tables into a single markdown table\n{image1}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../tests/data/multi-images.png"
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nDescribe the images.\n{image1}\n{image2}\n{image3}\n{image4}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../tests/data/image1.png",
    #         "image2": "../../tests/data/image2.png",
    #         "image3": "../../tests/data/image3.png",
    #         "image4": "../../tests/data/image4.png",
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nCombine information in both the tables into a single markdown table\n{image1}{image2}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../tests/data/table1.png",
    #         "image2": "../../tests/data/table2.png"
    #     }
    # },
]

pred = engine.infer_list(test_input)

print(pred)
print(pred[2][0][0].replace("｟newline｠", "\n"))
