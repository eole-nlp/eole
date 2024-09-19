# Serving models with Eole

The provided example configuration allows to serve Llama3-8B-Instruct.

```yaml
models_root: "." # used only for HF downloads for now, but might override $EOLE_MODEL_DIR at some point
models:
# local model
- id: "llama3-8b-instruct"
  path: "${EOLE_MODEL_DIR}/llama3-8b-instruct"
  preload: false
  config:
    quant_layers: ['gate_up_proj', 'down_proj', 'up_proj', 'linear_values', 'linear_query', 'linear_keys', 'final_linear']
    quant_type: "bnb_NF4"
# HF repo id, automatically downloaded to models_root
- id: "llama3-8b-instruct-hf"
  path: "fhdz/llama3-8b-instruct"
  preload: true
```

Note: the `preload` flag allow to load the corresponding model at server startup. See below for the two options.

## Retrieve and convert model

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

### Option 1 - Download and convert model

The first example `"llama3-8b-instruct"` requires you to manually convert the model in your desired `$EOLE_MODEL_DIR`.

```
eole convert HF --model_dir meta-llama/Meta-Llama-3-8B-Instruct --output $EOLE_MODEL_DIR/llama3-8b-instruct --token $HF_TOKEN
```

### Option 2 - Retrieve an already converted model from HF

The second example `"llama3-8b-instruct-hf"` downloads a model that has already been converted, for the sake of this example.

## Run server

```
eole serve -c serve.example.yaml
```

## Play with the API

FastAPI exposes a swagger UI by default. It should be accessible via your browser at `http://localhost:5000/docs`.