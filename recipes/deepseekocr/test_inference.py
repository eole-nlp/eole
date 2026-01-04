# flake8: noqa
import os
from rich import print
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY

mydir = os.getenv("EOLE_MODEL_DIR")

config = PredictConfig(
    model_path=os.path.join(mydir, "DeepSeek-OCR"),
    src="dummy",
    self_attn_backend="flash",
    max_length=8192,
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
    fuse_kvq=True,
    fuse_gate=True,
)

config.data_type = "image"
engine = InferenceEnginePY(config)

test_input = [
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

_, _, preds = engine.infer_list(test_input)

for i in range(len(test_input)):
    # [0] for first n_best
    print(f'{"#" * 40} example {i} {"#" * 40}')
    print(preds[i][0].replace("｟newline｠", "\n"))
