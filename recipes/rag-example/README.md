# Minimal RAG example

## Context

This recipe is intended as a minimal example for Retrieval Augmented Generation using Eole models.
It relies on additional tools, such as Langchain's [loaders](https://python.langchain.com/docs/integrations/document_loaders/pymupdf/) and [splitters](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/), as well as [ChromaDB](https://docs.trychroma.com/getting-started) for vector search.

The example is using the rather hard to digest ["EU AI Act" full text](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai#:~:text=The%20AI%20Act%20(Regulation%20(EU,regarding%20specific%20uses%20of%20AI.) for the sake of the exercise.

This is just a very quickly put together proof of concept, and is not expected to give perfect answers.

## Usage

### 0. Install requirements

```bash
pip install langchain pymupdf chromadb
```

### 1. Convert the model you want to use (Llama-3.1-8B by default)

**Set environment variables**

```bash
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

**Download and convert model**

```bash
eole convert HF --model_dir meta-llama/Meta-Llama-3.1-8B --output $EOLE_MODEL_DIR/llama3.1-8b --token $HF_TOKEN
```

### 2. Adapt and run the script

Modify, if needed, the model_path in `PredictConfig`:
```python
...
config = PredictConfig(
    model_path=os.path.expandvars("${EOLE_MODEL_DIR}/llama3.1-8b"), # <------ change if needed
    src="dummy",
    max_length=500,
...
```

**Run the script**
```bash
python3 test_rag.py
```

Note: You can test various queries by changing the `QUERY` variable.