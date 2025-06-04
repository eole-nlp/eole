# flake8: noqa

from rich import print
from eole.config.run import *
from eole.inference_engine import InferenceEnginePY

seed = 42

config = PredictConfig(
    model_path="./bagel",
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
    top_k=0,
    top_p=0.0,
    # top_p=0.8,
    # temperature=0.35,
    # beam_size=5,
    beam_size=1,
    # temperature=0.3,
    seed=seed,
    batch_size=1,
    batch_type="sents",
    self_attn_backend="pytorch",
    image_generation=True,
    image_width=1024,
    image_height=1024,
    # num_timesteps=10,
    num_timesteps=30,
    # num_timesteps=50,
    # self_attn_backend="flash", # not properly supported (mixed masking)
    output=f"generated_image_{seed}.png",
)

print(config)

# config.data_type = "image"
config.data_type = "text"
engine = InferenceEnginePY(config)

print(engine.predictor.model)
engine.predictor.model.count_parameters()

# prompt = "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere."

prompt = "A breathtaking photorealistic landscape of a windswept coastal cliff at golden hour. The scene features jagged rocks covered in moss, waves crashing below with mist rising, and seabirds flying overhead. The lighting is warm and natural, casting long shadows and reflecting on wet surfaces. The level of detail is ultra high, with textures of stone, water, and clouds rendered realistically, evoking a feeling of awe and solitude."

# test_input = [{
#     "text": f"<|im_start|>{prompt}<|im_end|><|im_start|>"
#     }] #not fully sure about prompt structure

test_input = [f"<|im_start|>{prompt}<|im_end|>"]

import torch
import numpy as np
import random

pred = engine.infer_list(test_input)

print(pred)
