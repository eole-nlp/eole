---
sidebar_position: 1
---

# Toolkit overview

Eole is a PyTorch toolkit for training, fine-tuning, evaluating, and serving
encoder, decoder, and encoder-decoder models. It originated from OpenNMT-py.

Start with the [README and installation instructions](index.md#installation),
the [quickstart](quickstart.md), or the [recipe index](recipes/README.md).
The reference sidebar is generated from the current Python configuration and API.

Current workflows include supported Hugging Face model conversion and direct
inference, LoRA and quantized fine-tuning, translation scoring, vision/OCR,
Whisper transcription, and an HTTP server with OpenAI-style and Anthropic-style
endpoints. Hardware, precision, and optional kernels determine which paths run.

Qwen3.8 native MTP inference currently supports single-sequence greedy text
requests. The recorded INT4/BF16 outputs differ from ordinary greedy decoding;
exact production token parity is unresolved. Its vision-language training class
loads MTP heads but does not implement their auxiliary training loss.
REINFORCE with scorer rewards is implemented; DPO, GRPO, and PPO are planned.
See the recipes for these limitations and what was actually tested.

## Citation

For work building on Eole's OpenNMT lineage, cite the
[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012):

```bibtex
@inproceedings{opennmt,
  author = {Guillaume Klein and Yoon Kim and Yuntian Deng and
            Jean Senellart and Alexander M. Rush},
  title = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proceedings of ACL},
  year = {2017},
  doi = {10.18653/v1/P17-4012}
}
```

Use [Discussions](https://github.com/eole-nlp/eole/discussions) for questions and
[Issues](https://github.com/eole-nlp/eole/issues) for bugs.
