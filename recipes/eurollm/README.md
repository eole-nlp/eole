# Serving Eurollm in a gradio interface with Eole

Run the following commands from the repository root. Install `gradio` and
`requests` for the translator interface. The server YAML selects GPU 0, BF16,
PyTorch attention, a 4096-token context, and greedy decoding. Allow roughly
18 GB for the 9B weights plus cache and working memory. Start the server before
launching Gradio; select the `EuroLLM-9B-Instruct` model in the interface.

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model


```
eole convert HF --model_dir utter-project/EuroLLM-9B-Instruct --output $EOLE_MODEL_DIR/EuroLLM-9B-Instruct --token $HF_TOKEN
```

## Run server with the config file from this folder (you can add options according to your needs)

```
eole serve -c recipes/eurollm/serve.yaml --host 127.0.0.1
```

See [the example `serve.yaml` file](serve.yaml).

## Start the gradio based translator

```
python apps/eole-translator.py
```

You can access the Web based translator from the url given by Gradio (either local or from Gradio proxy with share=True turned on)


## Alternatively you can also play with the API

FastAPI exposes a swagger UI by default. It should be accessible via your browser at `http://localhost:5000/docs`.
