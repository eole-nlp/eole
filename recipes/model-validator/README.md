# Model Validator

## 1. Edit the models you want to run in the `run.sh` script:

```bash
#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Define the models table
models=(
    "mistralai/Ministral-8B-Instruct-2410"
    "mistralai/Mistral-7B-v0.3"
    "mistralai/Mathstral-7B-v0.1"
    "meta-llama/Llama-3.2-1B"
    ...
)
...
```

## 2. Run the script

```bash
nohup ./run.sh > <custom_logfile_name>.log
```