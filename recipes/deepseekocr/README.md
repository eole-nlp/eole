# DeepSeek-OCR

### Set environment variables

```
export EOLE_MODEL_DIR=<where_to_store_models>
export HF_TOKEN=<your_hf_token>
```

## Convert the model

```
eole convert HF --model_dir deepseek-ai/DeepSeek-OCR --output $EOLE_MODEL_DIR/DeepSeek-OCR --token $HF_TOKEN
```

## Run the test script

```
python3 test_inference.py
```

This script shows the difference between two prompts and for two pages of the DeepSeek-OCR paper.


## Convert a PDF to markdown

The script is hardcoded with a path to a pdf file (the deepseek ocr paper stored locally)

```
python recipes/deepseekocr/pdf_ocr_mmd.py
```

This will spit out three files:
deepseekocr.mmd (simple markdown)
deepseekocr_det.mmd (markdown with coordinates)
deepseekocr_layouts.pdf (original pdf with boxings around text blocke, figures, tables)
