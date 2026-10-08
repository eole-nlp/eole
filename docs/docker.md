# Using Eole with Docker

Docker keeps Eole's Python and CUDA dependencies in a container. The host still
needs an NVIDIA driver compatible with the image's CUDA version, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Follow NVIDIA's installation and Docker runtime configuration instructions first.

## Choose or build an image

Choose a tag from the [published images](https://github.com/eole-nlp/eole/pkgs/container/eole).
Release images contain the source at that release: a 0.6.0 image does not include
later MTP or REINFORCE changes. For a particular checkout, run this from the
repository root:

```bash
docker build -t eole:local -f docker/Dockerfile \
  --build-arg TORCH_VERSION=2.10.0 \
  --build-arg CUDA_VERSION=12.8.0 .
export EOLE_IMAGE=eole:local
```

This uses the matching Eole base image from GHCR. If that base tag is unavailable,
build it locally first:

```bash
docker build \
  -t ghcr.io/eole-nlp/eole-base:torch2.10.0-ubuntu24.04-cuda12.8.0 \
  -f docker/Dockerfile-base \
  --build-arg TORCH_VERSION=2.10.0 \
  --build-arg CUDA_VERSION=12.8.0 .
```

The base Dockerfile installs PyTorch and FlashAttention 2.8.3, trying a community
prebuilt wheel before falling back to compilation. The full Dockerfile installs
Eole and `requirements.opt.txt`. Changing CUDA/PyTorch versions requires a
compatible base image and FlashAttention build. The `docker/build.sh` and
`docker/build_base.sh` scripts also **push images to GHCR**; use the commands above
for local builds.

For a published image, set `EOLE_IMAGE` to its full `ghcr.io/eole-nlp/eole:TAG`
name instead.

## Start a GPU container

Run from your checkout. Set `EOLE_MODEL_DIR` to an existing host model directory:

```bash
export EOLE_MODEL_DIR=/absolute/path/to/models
docker run --name eole-dev -it --gpus all --shm-size=8g \
  -v "$EOLE_MODEL_DIR:/models" \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e EOLE_MODEL_DIR=/models \
  -p 127.0.0.1:5000:5000 \
  --entrypoint /bin/bash "$EOLE_IMAGE"
```

The checkout is mounted at `/workspace`, and models at `/models`. Use these
**container paths** in YAML, not host paths. Mounted output files survive container
removal; files elsewhere remain in this container until it is removed. The image
runs as root by default, so newly created mounted files can be owned by root.
The shared-memory allocation can be adjusted for training data workers.

Inside the container, check GPU access and install the mounted checkout:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install "setuptools<69" wheel packaging ninja psutil
MAX_JOBS=2 python -m pip install -e /workspace --no-build-isolation
python -c 'from eole import _ops; print("Eole CUDA extension loaded")'
```

A normal `docker build` does not expose the GPU. Eole's current `setup.py` skips
its CUDA extension when `torch.cuda.is_available()` is false, even when `nvcc` is
installed. The installation above builds it with the GPU visible. Alternatively,
for direct script use, run `MAX_JOBS=2 python setup.py build_ext --inplace` from
`/workspace`. Rebuild when changing GPU architecture, PyTorch, or CUDA.

For Qwen gated-delta models, check the additional kernel imports:

```bash
python -m pip install fla-core
python -c 'from flash_attn import flash_attn_func, flash_attn_with_kvcache'
python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule'
```

See the [installation instructions](../README.md#installation) for FlashAttention
source installation and optional `causal-conv1d`. Successful imports do not prove
that every kernel executes on your GPU.

## Run inference or serve with YAML

Inside the container, use the same YAML recipes as a native installation. For a
converted Qwen3.8 checkpoint stored in the mounted model directory:

```bash
export QWEN38_MODEL=/models/Qwen3.8-27B-INT4
python -m eole.bin.main predict -c recipes/qwen38/predict.yaml
python -m eole.bin.main serve -c recipes/qwen38/serve.yaml \
  --host 0.0.0.0 --port 5000
```

The server must bind to `0.0.0.0` inside the container for Docker port forwarding.
The example publishes only to the host's loopback address. From a host terminal:

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/v1/models
```

See the [Qwen3.8 recipe](../recipes/qwen38/README.md) for checkpoint conversion,
MTP requirements, and validation, the [server recipe](../recipes/server/README.md)
for API examples, and the [Claude Code recipe](../recipes/claude-code/README.md)
for connecting a host client to the endpoint. Training works the same way:
`python -m eole.bin.main train -c /workspace/path/to/train.yaml`, with datasets and
checkpoint output directories on mounted storage.

Exit the shell to stop the container, then resume it with `docker start -ai
eole-dev`. To discard its container-local changes, use `docker rm eole-dev` after
stopping it. A new container needs the runtime kernel build again unless those
artifacts are retained in the mounted checkout or baked into a separately saved
image.
