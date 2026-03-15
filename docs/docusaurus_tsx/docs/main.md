---
sidebar_position: 2
---

# Overview


This portal provides detailed documentation of the EOLE toolkit — an open language modeling toolkit based on PyTorch, initially spun off from OpenNMT-py.

If you need a step-by-step overview, please read the [Quickstart](quickstart).


## Installation

### Using Docker

To facilitate setup and reproducibility, Docker images are available via the GitHub Container Registry: [EOLE Docker Images](https://github.com/eole-nlp/eole/pkgs/container/eole).

To pull the latest image:
```bash
docker pull ghcr.io/eole-nlp/eole:latest
```

Example one-liner to run a container and open a bash shell:
```bash
docker run --rm -it --runtime=nvidia ghcr.io/eole-nlp/eole:latest
```

> **Note**: Ensure you have the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed to take advantage of CUDA/GPU features.

### Installing Locally

#### Requirements

- Python >= 3.10
- PyTorch >= 2.8

#### Installation from Source

```bash
git clone https://github.com/eole-nlp/eole
cd eole
pip install -e .
```

*(Optional)* Some advanced features (e.g. pretrained models or specific transforms) require extra packages:
```bash
pip install -r requirements.opt.txt
```

#### Manual Installation of Some Dependencies

##### Flash Attention

```bash
pip install flash-attn --no-build-isolation
```

##### AWQ

```bash
pip install autoawq
```

And you are ready to go!

Take a look at the [quickstart](quickstart) to familiarize yourself with the main training and inference workflow.

## Citation

When using EOLE for research, please cite our
[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

## Additional resources

* [GitHub Discussions](https://github.com/eole-nlp/eole/discussions)
* [GitHub Issues](https://github.com/eole-nlp/eole/issues)
* [Recipes](https://github.com/eole-nlp/eole/tree/main/recipes)

