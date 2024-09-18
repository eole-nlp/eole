# Serving models with Eole

The provided example configuration allows to serve Llama3-8B-Instruct.

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Download and convert model

```
eole convert HF --model_dir meta-llama/Meta-Llama-3-8B-Instruct --output $EOLE_MODEL_DIR/llama3-8b-instruct --token $HF_TOKEN
```

## Run server

```
eole serve -c serve.example.yaml
```

## Play with the API

FastAPI exposes a swagger UI by default. It should be accessible via your browser at `http://localhost:5000/docs`.