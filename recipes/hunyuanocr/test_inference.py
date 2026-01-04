# flake8: noqa
import os
from rich import print


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(
        model_path=os.path.join(mydir, "HunyuanOCR"),
        src="dummy",
        self_attn_backend="flash",
        max_length=4096,
        world_size=1,
        gpu_ranks=[0],
        parallel_mode="data_parallel",
        compute_dtype="bf16",
        top_k=1.0,
        top_p=0.0,
        temperature=1.0,
        beam_size=1,
        seed=42,
        batch_size=4,
        batch_type="sents",
        report_time=True,
        fuse_gate=True,
        fuse_kvq=True,
    )

    config.data_type = "image"
    return config


def build_test_inputs():
    return [
        {
            "text": (
                "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>{image}"
                "Extract all information from the main body of the document image "
                "and represent it in markdown format, ignoring headers and footers. "
                "Tables should be expressed in HTML format, formulas in the document "
                "should be represented using LaTeX format, and the parsing should be "
                "organized according to the reading order.<｜hy_User｜>"
            ),
            "images": {"image": "eole/tests/data/images/deepseekpapertable.png"},
        },
        {
            "text": (
                "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>{image}"
                "提 取 文 档 图 片 中 正 文 的 所 有 信 息用markdown 格 式 表 示 ， "
                "其 中 页眉、页脚部分忽略，表格用html 格式表达，文档中公式用LATEX格式表示，"
                "按照阅读顺序组织进行解析。<｜hy_User｜>"
            ),
            "images": {"image": "eole/tests/data/images/deepseekpapertable.png"},
        },
        {
            "text": (
                "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>{image}"
                "Detect and recognize text in the image, and output the text "
                "coordinates in a formatted manner.<｜hy_User｜>"
            ),
            "images": {"image": "eole/tests/data/images/deepseekpaper.png"},
        },
        {
            "text": (
                "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>{image}"
                "检测并识别图片中的文字，将文本坐标格式化输出。<｜hy_User｜>"
            ),
            "images": {"image": "eole/tests/data/images/deepseekpaper.png"},
        },
    ]


def postprocess_and_print(pred, test_input):

    for i in range(len(test_input)):
        print(f'{"#" * 40} example {i} {"#" * 40}')
        text = pred[i][0]
        text = text.replace("<｜hy_place▁holder▁no▁110｜>", "<quad>")
        text = text.replace("<｜hy_place▁holder▁no▁111｜>", "</quad>")
        text = text.replace("<｜hy_place▁holder▁no▁112｜>", "<ref>")
        text = text.replace("<｜hy_place▁holder▁no▁113｜>", "</ref>")
        text = text.replace("<｜hy_place▁holder▁no▁114｜>", "<pFig>")
        text = text.replace("<｜hy_place▁holder▁no▁115｜>", "</pFig>")
        print(text.replace("｟newline｠", "\n"))


def main():
    from eole.inference_engine import InferenceEnginePY

    config = build_config()
    engine = InferenceEnginePY(config)

    test_input = build_test_inputs()
    try:
        _, _, pred = engine.infer_list(test_input)

        postprocess_and_print(pred, test_input)

    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
