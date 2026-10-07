# DeepSeek-OCR

Run from the repository root in a CUDA-enabled Eole environment. The examples
use BF16, GPU 0, and FlashAttention; install `flash-attn --no-build-isolation` or
select the PyTorch attention backend in the configuration.

```bash
export EOLE_MODEL_DIR=/path/to/models
eole convert HF --model_dir deepseek-ai/DeepSeek-OCR \
  --output "$EOLE_MODEL_DIR/DeepSeek-OCR" --token "$HF_TOKEN"
python recipes/deepseekocr/test_inference.py
```

The script prints OCR and grounded Markdown outputs for two bundled images in
`eole/tests/data/images`. Set `HF_TOKEN` if authentication is required.

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
