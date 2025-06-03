# flake8: noqa

from rich import print
from eole.config.run import *
from eole.inference_engine import InferenceEnginePY

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
    seed=42,
    batch_size=1,
    batch_type="sents",
    self_attn_backend="pytorch",
    image_generation=True,
    image_width=1024,
    image_height=1024,
    # self_attn_backend="flash", # not properly supported (mixed masking)
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

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pred = engine.infer_list(test_input)

print(pred)

exit()

test_input = [
    {
    #    "text": "<|im_start|>List the top 5 countries in Europe with the highest GDP from this image<|im_end|>\n{image1}\n",
       "text": "{image1}<|im_start|>List the top 5 countries in Europe with the highest GDP from this image<|im_end|><|im_start|>",
    #    "text": "{image1}", # replicate first pass of bagel with image only
       "images": {"image1": "../../eole/tests/data/images/gdp.png"},
    },
    # {
    #     # "text": "{image1}<|im_start|>When did things start to go wrong for dark dragon?<|im_end|>",
    #     # "text": "{image1}",
    #     # "text": "{image1}<|im_start|>Describe<|im_end|><|im_start|>", # bagel weirdly starts decoding by adding a <|im_start|> token
    #     "text": "{image1}<|im_start|>When did things start to go wrong for dark dragon?<|im_end|><|im_start|>",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/loss_curve.jpg"
    #     }
    # },
    # {
    #     "text": "{image1}<|im_start|>Which model is best?<|im_end|>",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/loss_curve.jpg"
    #     }
    # },
    # {
    #     "text": "{image1}<|im_start|>Can someone explain what’s funny about this meme??<|im_end|>",
    #     "images": {
    #         "image1": "./BAGEL/test_images/meme.jpg"
    #     }
    # }
    # {
    #     "text": "{image1}<|im_start|>Is this person really big, or is this building just super small?<|im_end|>",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/pisa_2.jpg"
    #     }
    # },
    # {
    #     "text": "<start_of_turn>user\nCombine information in both the tables into a single markdown table\n{image1}\n{image2}<end_of_turn><start_of_turn>model\n",
    #     "images": {
    #         "image1": "../../eole/tests/data/images/table1.png",
    #         "image2": "../../eole/tests/data/images/table2.png",
    #     },
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

import torch
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pred = engine.infer_list(test_input)

print(pred)
print(pred[2][0][0].replace("｟newline｠", "\n"))
