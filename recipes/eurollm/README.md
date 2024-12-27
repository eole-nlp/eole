# Serving Eurollm in a gradio interface with Eole

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
eole serve -c serve.yaml
```

## Start the gradio based translator

```
python eole-translator.py
```

You can access the Web based translator from the url given by Gradio (either local or from Gradio proxy with share=True turned on)


## Alternatively you can also play with the API

FastAPI exposes a swagger UI by default. It should be accessible via your browser at `http://localhost:5000/docs`.
