# flake8: noqa
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode

"""
CompilationMode:
    NONE = 0
    No torch.compile compilation is applied, model runs in fully eager pytorch mode.
    The model runs as-is.
    STOCK_TORCH_COMPILE = 1
    The standard `torch.compile` compilation pipeline.
    DYNAMO_TRACE_ONCE = 2
    Single Dynamo trace through the model, avoiding recompilation.
    VLLM_COMPILE = 3
    Custom vLLM Inductor-based backend with caching, piecewise compilation,
    shape specialization, and custom passes.

CUDAGraphMode:
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)
"""
compilation_config = {"mode": 3, "cudagraph_mode": "NONE"}


def main():
    # Initialize the model with a custom max_model_len
    llm = LLM(
        model="google/gemma-3-1b-it",
        max_model_len=2048,  # <-- Set the total context length here (prompt + generated)
        gpu_memory_utilization=0.95,
        compilation_config=compilation_config,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048)  # how many tokens to generate

    # Your prompts
    prompts = [
        "<start_of_turn>user\nGenerate a 200 word text talking about George Orwell.<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWhat is the meaning of life?<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWho is Elon Musk?<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nWhat is beyond the milky way?<end_of_turn>\n<start_of_turn>model\n",
    ]

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
