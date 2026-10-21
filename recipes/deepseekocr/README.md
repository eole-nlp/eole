# DeepSeek-OCR

Run from the repository root in a CUDA-enabled Eole environment. The examples
use BF16, GPU 0, and FlashAttention; install `flash-attn --no-build-isolation` or
select the PyTorch attention backend in the configuration.

## Prepare the checkpoint

With Eole 0.6.4 or newer, choose either preparation option below. Both use the
same directory expected by the scripts and YAMLs:

```bash
export EOLE_MODEL_DIR=/path/to/models
```

**Download pre-converted weights:** the public
[Eole DeepSeek-OCR package](https://huggingface.co/eole-nlp/DeepSeek-OCR-eole)
includes converted weights, configuration and tokenizer. No conversion or HF
token is required:

```bash
hf download eole-nlp/DeepSeek-OCR-eole --local-dir "$EOLE_MODEL_DIR/DeepSeek-OCR"
```

**Or convert the original HF checkpoint:** the converter accepts the upstream
repository ID or an existing local HF directory:

```bash
eole convert HF --model_dir deepseek-ai/DeepSeek-OCR \
  --output "$EOLE_MODEL_DIR/DeepSeek-OCR" --dtype bf16
```

Use a new destination for either option. Add `--token "$HF_TOKEN"` to conversion
if authentication is required. To pin a pre-converted download, add
`--revision COMMIT_SHA`; validated pins are in the
[RunPod model catalog](../runpod/models.json). Downloading pre-converted weights
does not remove runtime kernel or GPU memory requirements.

After either preparation option:

```bash
python recipes/deepseekocr/test_inference.py
```

The script prints OCR and grounded Markdown outputs for two bundled images in
`eole/tests/data/images`.

## Convert a PDF to Markdown

Install the PDF helper dependencies and supply your own PDF and output directory:

```bash
pip install pymupdf img2pdf
python recipes/deepseekocr/pdf_ocr_mmd.py \
  -c recipes/deepseekocr/predict-pdf.yaml \
  --input /path/to/document.pdf --output-dir ./ocr-output
```

The helper renders pages to images, runs OCR, and writes `document.mmd`,
`document_det.mmd` (grounding coordinates), and `document_layouts.pdf` (annotated
pages). Batch size defaults to one; increase it in the YAML only if VRAM permits.
All pages are currently rendered into host memory, so split very large PDFs.
