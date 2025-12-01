# flake8: noqa
import os
from rich import print
from eole.config.run import *
from eole.inference_engine import InferenceEnginePY

mydir = os.getenv("EOLE_MODEL_DIR")

config = PredictConfig(
    model_path=os.path.join(mydir, "HunyuanOCR"),
    src="dummy",
    self_attn_backend="flash",
    max_length=4096,
    gpu_ranks=[0],
    compute_dtype="bf16",
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    beam_size=1,
    seed=42,
    batch_size=1,
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
        "text": "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>{image} Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order. Make sure you do not repeat sentences.<｜hy_User｜>",
        "images": {"image": "eole/tests/data/images/deepseekpapertable.png"},
    },
    {
        "text": "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>{image}提 取 文 档 图 片 中 正 文 的 所 有 信 息用markdown 格 式 表 示 ， 其 中 页眉、页脚部分忽略，表格用html 格式表达，文档中公式用LATEX格式表示，按照阅读顺序组织进行解析。<｜hy_User｜>",
        "images": {"image": "eole/tests/data/images/deepseekpapertable.png"},
    },
]

pred = engine.infer_list(test_input)

print(pred, "\n\n")

text = pred[2][0][0]
text = text.replace("<｜hy_place▁holder▁no▁110｜>", "<quad>")
text = text.replace("<｜hy_place▁holder▁no▁111｜>", "</quad>")
text = text.replace("<｜hy_place▁holder▁no▁112｜>", "<ref>")
text = text.replace("<｜hy_place▁holder▁no▁113｜>", "</ref>")
text = text.replace("<｜hy_place▁holder▁no▁114｜>", "<pFig>")
text = text.replace("<｜hy_place▁holder▁no▁115｜>", "</pFig>")
print(text.replace("｟newline｠", "\n"))
print("\n\n")
text = pred[2][1][0]
text = text.replace("<｜hy_place▁holder▁no▁110｜>", "<quad>")
text = text.replace("<｜hy_place▁holder▁no▁111｜>", "</quad>")
text = text.replace("<｜hy_place▁holder▁no▁112｜>", "<ref>")
text = text.replace("<｜hy_place▁holder▁no▁113｜>", "</ref>")
text = text.replace("<｜hy_place▁holder▁no▁114｜>", "<pFig>")
text = text.replace("<｜hy_place▁holder▁no▁115｜>", "</pFig>")
print(text.replace("｟newline｠", "\n"))
