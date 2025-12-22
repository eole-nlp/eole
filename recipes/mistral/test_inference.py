# flake8: noqa
import os
from rich import print
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY

sys_prmt = "[SYSTEM_PROMPT]# HOW YOU SHOULD THINK AND ANSWER\n\nFirst draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response.[/SYSTEM_PROMPT]"

mydir = os.getenv("EOLE_MODEL_DIR")

config = PredictConfig(
    model_path=os.path.join(mydir, "mistralai/Ministral-3-14B-Reasoning-2512"),
    # model_path=os.path.join(mydir, "mistralai/mistral-3.1-24B-instruct"),
    src="dummy",
    max_length=4096,
    gpu_ranks=[0],
    # quant_type="bnb_NF4",
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
)

# print(config)

config.data_type = "image"
engine = InferenceEnginePY(config)

# print(engine.predictor.model)
# engine.predictor.model.count_parameters()

test_input = [
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

pred = engine.infer_list(test_input)

print(pred)
for i in range(len(test_input)):
    print(
        pred[2][i][0]
        .replace("[THINK]", "\[THINK]\n\n")
        .replace("[/THINK]", "\[/THINK]\n\n")
        .replace("｟newline｠", "\n")
    )
    print("\n\n")
