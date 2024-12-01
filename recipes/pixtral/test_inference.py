# flake8: noqa

from rich import print
from eole.config.run import *
from eole.inference_engine import InferenceEnginePY

config = PredictConfig(
    model_path="./pixtral-12b",
    src="dummy",
    max_length=500,
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
    compute_dtype="fp16",
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
    {
        "text": "<s>[INST]List the top 5 countries in Europe with the highest GDP\n{image1}[/INST]",
        "images": {"image1": "./test_data/gdp.png"},
    },
    # {
    #     "text": "[INST]When did things start to go wrong for dark dragon?\n{image1}[/INST]",
    #     "images": {
    #         "image1": "./test_data/loss_curve.jpg"
    #     }
    # },
    # {
    #     "text": "<s>[INST]Is this person really big, or is this building just super small?\n{image1}[/INST]",
    #     "images": {
    #         "image1": "./test_data/pisa_2.jpg"
    #     }
    # },
    # {
    #     "text": "<s>[INST]Combine information in both the tables into a single markdown table\n{image1}\n{image2}[/INST]",
    #     "images": {
    #         "image1": "./test_data/table1.png",
    #         "image2": "./test_data/table2.png"
    #     }
    # },
    # {
    #     "text": "<s>[INST]Combine information in both the tables into a single markdown table\n{image1}[/INST]",
    #     "images": {
    #         "image1": "./test_data/multi-images.png"
    #     }
    # },
    # {
    #     "text": "<s>[INST]Describe the images.\n{image1}\n{image2}\n{image3}\n{image4}[/INST]",
    #     "images": {
    #         "image1": "./test_data/image1.png",
    #         "image2": "./test_data/image2.png",
    #         "image3": "./test_data/image3.png",
    #         "image4": "./test_data/image4.png",
    #     }
    # },
    # {
    #     "text": "<s>[INST]Combine information in both the tables into a single markdown table\n{image1}{image2}[/INST]",
    #     "images": {
    #         "image1": "./test_data/table1.png",
    #         "image2": "./test_data/table2.png"
    #     }
    # },
]

pred = engine.infer_list(test_input)

print(pred)
print(pred[2][0][0].replace("｟newline｠", "\n"))
