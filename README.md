# EOLE

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://eole-nlp.github.io/eole)

Open language modeling toolkit based on [PyTorch](https://pytorch.org).

## 👷‍♂️🚧 Work in Progress

[EOLE](https://github.com/eole-nlp/eole) is a spin-off of the [OpenNMT-py](https://github.com/opennmt/opennmt-py) project. We aim to maintain the research-friendly approach of the original project while updating the structure and expanding it to include new topics related to large language models (LLMs) and various other techniques. Our goal is to provide a comprehensive yet compact and modular codebase for experimenting with various types of language models (encoder, decoder, seq2seq).

---

### Current State

We have made significant progress in several areas:

- **Configuration Management**: Streamlined through [pydantic](https://docs.pydantic.dev) models.
- **Command Line Entry Points**: Improved using structured subparsers for better organization.
- **Reproducible Recipes**: Provided for widely used models and tasks, ensuring consistency and reliability.
- **Core API Simplification**: Refined around the new configuration objects for ease of use.

### Future Directions

There are still several exciting avenues to explore:

- **Further Simplification and Refactoring**: Continue enhancing the codebase for clarity and efficiency.
- **Inference Server**: Develop a robust solution for model inference.
- **Additional Recipes**: Expand the library of reproducible recipes.
- **Documentation**: Enhance and expand the documentation for better user guidance.
- **Test Coverage**: Improve testing to ensure code reliability and performance.
- **Logging Enhancements**: Implement more sophisticated logging mechanisms.
- **Broader Model Support**: Extend support to include a wider range of open models, potentially multi-modal.

---

## Key Features

- **Versatile Training and Inference**: Train from scratch, finetune, and infer models of various architectures including Transformer Encoder/Decoder/EncoderDecoder and RNN EncoderDecoder.
- **Dynamic Data Transforms**: Apply on-the-fly transformations in the dataloading logic for both training and inference.
- **Comprehensive LLM Support**: Includes converters for Llama, Mistral, Phi, OpenLlama, Redpajama, MPT-7B, and Falcon models.
- **Advanced Quantization**: Support for 8-bit and 4-bit quantization, along with LoRA adapters, with or without checkpointing, as well as mixed precision (FP16).
- **Efficient Finetuning**: Finetune 7B and 13B models on a single RTX 24GB GPU using 4-bit quantization.
- **Flexible Inference**: Perform inference in 4-bit or 8-bit using the same layer quantization methods as in finetuning.
- **Tensor Parallelism**: Enable tensor parallelism for both training and inference when models exceed the memory capacity of a single GPU.

---

## Setup

### Using Docker

To facilitate setup and reproducibility, we provide Docker images via the GitHub Container Registry: [EOLE Docker Images](https://github.com/eole-nlp/eole/pkgs/container/eole).

You can customize the workflow and build your own images based on specific needs using `build.sh` and `Dockerfile` in the `docker` directory of the repository.

To pull the Docker image:
```bash
docker pull ghcr.io/eole-nlp/eole:0.0.1-ubuntu22.04-cuda12.1
```

Example one-liner to run a container and open a bash shell within it:
```bash
docker run --rm -it --runtime=nvidia ghcr.io/eole-nlp/eole:0.0.1-ubuntu22.04-cuda12.1
```

> **Note**: Ensure you have the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (formerly nvidia-docker) installed to take advantage of CUDA/GPU features.

Depending on your needs, you can add various flags:
- `-p 5000:5000`: Forward an exposed port from your container to your host.
- `-v /some/local/directory:/some/container/directory`: Mount a local directory to a container directory.
- `--entrypoint some_command`: Run a specific command as the container entry point (instead of the default bash shell).

### Installing Locally

#### Requirements

- Python >= 3.10
- PyTorch >= 2.3 < 2.4

#### Installation from Source

To install from source:
```bash
git clone https://github.com/eole-nlp/eole
cd eole
pip install -e .
```

#### Installation from PyPI

Installation from PyPI will be available soon.

#### Notes

If you encounter a `MemoryError` during installation, try using `pip` with the `--no-cache-dir` option.

(Optional) Some advanced features (e.g., pretrained models or specific transforms) require extra packages. Install them with:
```bash
pip install -r requirements.opt.txt
```

### Manual Installation of Some Dependencies

#### Apex

Apex is recommended for improved performance, especially for the legacy fusedadam optimizer and FusedRMSNorm.
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --no-build-isolation --config-settings --build-option="--cpp_ext --cuda_ext --deprecated_fused_adam --xentropy --fast_multihead_attn" ./
cd ..
```

#### Flash Attention

As of October 2023, Flash Attention 1 has been integrated into PyTorch v2, but using Flash Attention 2 with v2.3.1 is recommended for sliding window attention support. 

For regular `position_encoding=True` or Rotary with `max_relative_positions=-1`, `eole` will attempt to use an optimized dot-product path. To use [Flash Attention](https://github.com/Dao-AILab/flash-attention#installation-and-features), install it manually:
```bash
pip install flash-attn --no-build-isolation
```

If Flash Attention 2 is not installed, `F.scaled_dot_product_attention` from PyTorch 2.x will be used.

For `max_relative_positions > 0` or Alibi `max_relative_positions=-2`, `eole` will use legacy code for matrix multiplications. Flash Attention and `F.scaled_dot_product_attention` offer faster performance and better GPU memory efficiency.

#### AWQ

For inference or quantizing an AWQ model, AutoAWQ is required. Install it with:
```bash
pip install autoawq
```

For more details, refer to [AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

---

## Contributing

We love contributions! Please look at issues marked with the [contributions welcome](https://github.com/eole-nlp/eole/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tag.

Before raising an issue, make sure you read the requirements and the [Full Documentation](https://eole-nlp.github.io/eole). You can also check if a [Recipe](https://github.com/eole-nlp/eole/tree/main/recipes) fits your use case.

Unless there is a bug, please use the [Discussions](https://github.com/eole-nlp/eole/discussions) tab to ask questions or propose new topics/features.