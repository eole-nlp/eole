# Compile inference with torch.compile

Set compilation environment variables before importing Eole or launching a run:

```bash
export EOLE_TORCH_COMPILE=1
export EOLE_COMPILE_MODE=2
eole predict -c inference.yaml
```

| Mode | Compilation boundary | CUDA graphs |
|---|---|---|
| `0` | Decoder | Enabled where supported |
| `1` | Decoder | Disabled |
| `2` | Decoder layers | Enabled where supported |
| `3` | Decoder layers | Disabled |

Compilation and warmup latency vary with model, backend, shapes, and hardware.
Measure cold setup separately from warm generation; compilation is not a
universal speedup. See the [compilation guide](https://github.com/eole-nlp/eole/blob/main/TORCHCOMPILE_README.md)
for supported paths and diagnostic settings.

## Python inference API

```python
from eole.config.run import PredictConfig
from eole.inference_engine import InferenceEnginePY

config = PredictConfig(
    model_path="/path/to/converted/model",
    src="dummy",
    gpu_ranks=[0],
    compute_dtype="bf16",
    beam_size=1,
    top_k=1,
)
engine = InferenceEnginePY(config)
try:
    scores, estimates, predictions = engine.infer_list(["Hello, world!"])
    print(predictions[0][0])
    # Streaming takes one string and yields decoded text chunks.
    for chunk in engine.infer_list_stream("Hello, world!"):
        print(chunk, end="", flush=True)
finally:
    engine.terminate()
```

Use the model's expected chat template for chat prompts. The example demonstrates
the return contract; it does not download the placeholder model path.
