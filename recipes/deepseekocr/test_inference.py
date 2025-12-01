# flake8: noqa
import os
from rich import print
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY

mydir = os.getenv("EOLE_MODEL_DIR")

config = PredictConfig(
    model_path=os.path.join(mydir, "DeepSeek-OCR"),
    src="dummy",
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
    fuse_kvq=False,
    fuse_gate=True,
    # block_ngram_repeat=5,
)
# print(config)

config.data_type = "image"
engine = InferenceEnginePY(config)

# print(engine.predictor.model)
# engine.predictor.model.count_parameters()

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

pred = engine.infer_list(test_input)

print(pred[2][0][0].replace("｟newline｠", "\n"))
print("\n\n")
print(pred[2][1][0].replace("｟newline｠", "\n"))
print("\n\n")
print(pred[2][2][0].replace("｟newline｠", "\n"))
print("\n\n")
print(pred[2][3][0].replace("｟newline｠", "\n"))
