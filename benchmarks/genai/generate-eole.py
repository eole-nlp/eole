# flake8: noqa
import os
from rich import print


def build_config():
    from eole.config.run import PredictConfig

    mydir = os.getenv("EOLE_MODEL_DIR")
    if mydir is None:
        raise RuntimeError("EOLE_MODEL_DIR environment variable is not set")

    config = PredictConfig(
        model_path=os.path.join(mydir, "gemma-3-1b-it"),
        src="dummy",
        self_attn_backend="flash",
        max_length=2048,
        world_size=1,
        gpu_ranks=[0],
        parallel_mode="data_parallel",
        compute_dtype="bf16",
        top_k=0.0,
        top_p=0.0,
        temperature=1.0,
        beam_size=1,
        batch_size=4,
        batch_type="sents",
        report_time=True,
        fuse_gate=True,
        fuse_kvq=True,
    )

    return config


def build_test_inputs():
    return [
        "<start_of_turn>user\nGenerate a 200 word text talking about George Orwell.<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWhat is the meaning of life?<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWho is Elon Musk?<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWhat is beyond the milky way?<end_of_turn>\n<start_of_turn>model\n",
    ]


def postprocess_and_print(pred, test_input):

    for i in range(len(test_input)):
        print(f'{"#" * 40} example {i} {"#" * 40}')
        text = pred[i][0]
        print(text.replace("｟newline｠", "\n"))


def main():
    from eole.inference_engine import InferenceEnginePY

    config = build_config()
    engine = InferenceEnginePY(config)

    test_input = build_test_inputs()
    try:
        _, _, pred = engine.infer_list(test_input)

        # postprocess_and_print(pred, test_input)

    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
