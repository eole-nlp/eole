# Hunyuan-MT-7B

The Eole local workspace uses Tencent's **Hy-MT2-7B** by default for
translation. Follow the [Translate workspace recipe](../translation/README.md)
to download the checkpoint and start the browser UI.

To serve the original [Hunyuan-MT-7B checkpoint](https://huggingface.co/tencent/Hunyuan-MT-7B)
through the API, use [serve.yaml](serve.yaml). From the repository root, set your
model directory and convert the checkpoint if needed.

```bash
export EOLE_MODEL_DIR=/path/to/models
eole convert HF --model_dir tencent/Hunyuan-MT-7B \
  --output "$EOLE_MODEL_DIR/hunyuan-7B/merged" --dtype bf16
eole serve -c recipes/hunyuan/serve.yaml --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000/docs` to try the API with model ID
`Hunyuan-MT-7B-eole`. The model card provides prompts for the language pair you
want to translate.
