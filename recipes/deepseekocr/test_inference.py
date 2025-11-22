# flake8: noqa

from rich import print
from eole.config.run import *
from eole.inference_engine import InferenceEnginePY

config = PredictConfig(
    model_path="/mnt/InternalCrucial4/LLM_work/deepseek-ocr",
    src="dummy",
    max_length=4096,
    gpu_ranks=[0],
    compute_dtype="bf16",
    top_p=0.0,
    top_k=0,
    temperature=0.0,
    beam_size=1,
    seed=12,
    batch_size=1,
    batch_type="sents",
    report_time=True,
    block_ngram_repeat=5,
)

print(config)

config.data_type = "image"
engine = InferenceEnginePY(config)

print(engine.predictor.model)
engine.predictor.model.count_parameters()
test_input = [
    {
        "text": "{image}\n<|grounding|>Convert the document to markdown.",
        "images": {"image": "/mnt/InternalCrucial4/LLM_work/deepseek-ocr/deepseekpaper.png"},
    },
]
"""
test_input = [
    {
        "text": "{image}\nFree OCR.",
        "images": {"image": "/mnt/InternalCrucial4/LLM_work/deepseek-ocr/boursobank.png"},
    },
]
"""
pred = engine.infer_list(test_input)

print(pred)
print(pred[2][0][0].replace("｟newline｠", "\n"))
