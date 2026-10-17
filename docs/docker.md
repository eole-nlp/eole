# Using Eole with Docker

The host needs Docker, a compatible NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Configure the NVIDIA Docker runtime before starting a GPU container.

## Build or choose an image

The tested default is Ubuntu 24.04 / Python 3.12, PyTorch **2.14.1+cu132**,
CUDA **13.2.1**, FlashAttention **2.8.3**, and FLA **0.5.2**. The base upgrades
nvcc to **13.2.86**: the original CUDA 13.2.1 image has an older compiler.
PyTorch wheels, CUDA toolkit and FlashAttention wheels must be compatible; do not
substitute the newest CUDA toolkit independently of the PyTorch wheel index.
For a different toolkit family, `docker/build_base.sh` disables the CUDA13.2
compiler override; `CUDA_NVCC_PACKAGE` can explicitly select a compiler package.

Build the base first, then the full image, from the repository root:

```bash
docker/build_base.sh 2.14.1 13.2.1
docker build -t eole:local -f docker/Dockerfile .
export EOLE_IMAGE=eole:local
```

Both commands build locally. Nothing is pushed. The full Dockerfile uses the
matching base tag; a newly selected combination must be built first or already
exist on GHCR. Both Dockerfiles accept `TORCH_VERSION` and `CUDA_VERSION` build
arguments. Full image tags now include the complete CUDA patch version.

Alternatively, choose an existing tag from the
[published images](https://github.com/eole-nlp/eole/pkgs/container/eole) and set
`EOLE_IMAGE=ghcr.io/eole-nlp/eole:TAG`. Older published 0.6.0 images do not contain
the current workspace or all its dependencies; updating these Dockerfiles does
not update existing registry tags.

The base tries a community FlashAttention wheel before compiling from source.
The full image includes `requirements.opt.txt`, Transformers, FLA and the
`ui,translation,ocr,rag` extras. FFmpeg provides the shared libraries required by
TorchCodec audio/video decoding. `.dockerignore` excludes weights, build caches
and host CUDA binaries. Use a clean checkout to avoid copying unrelated local
files into your image.

## Start a GPU container and build Eole kernels

```bash
export EOLE_MODEL_DIR=/absolute/path/to/models
docker run --name eole-dev -it --gpus all --shm-size=8g \
  -v "$EOLE_MODEL_DIR:/models" \
  -v "$PWD/outputs:/outputs" \
  -p 127.0.0.1:5000:5000 "$EOLE_IMAGE"
```

The default shell starts in `/eole`, the image's own source tree. Inside it:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
eole-build-kernels
python -c 'from flash_attn import flash_attn_func, flash_attn_with_kvcache'
python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule'
```

A normal Docker image build and GitHub-hosted runner have no visible GPU.
Eole's setup skips its CUDA extension in that case. `eole-build-kernels` installs
Eole without build isolation **inside the running GPU container**, targeting all
visible GPU architectures. This enables Marlin INT4 and Eole's CUDA kernels.
Rebuild after changing GPU architectures, PyTorch or CUDA. Imports alone do not
prove every kernel executes on your hardware.

The helper operates on `/eole`, not a mounted host checkout: it does not overwrite
native host CUDA binaries. Compilation may take several minutes. The artifacts
persist in this container across `docker stop` / `docker start -ai eole-dev`.
A fresh container needs the build again. To save a locally reusable GPU-specific
image, stop the container and run `docker commit eole-dev eole:local-kernels`.
Do not publish such an image as supporting architectures you did not build/test.

Use container model/data/output paths in YAML. Some converted checkpoints store
absolute tokenizer paths in `config.json`; update those paths in a copy of the
config, or mount the models at the same absolute path inside the container.
Files under `/outputs` survive container deletion. Files elsewhere remain in
this container until it is removed. The image runs as root, so mounted output
files may be owned by root. Adjust shared memory for training data workers.

For development, mount a **separate clean clone** at `/workspace` and install it
with `MAX_JOBS=2 python -m pip install -e /workspace --no-build-isolation`.
This writes a container-specific extension into that clone; avoid using your
native Python checkout for this purpose.

## Run inference or serve with YAML

Inside the container, for a converted RedHat Qwen3.8 checkpoint:

```bash
export QWEN38_MODEL=/models/Qwen3.8-27B-INT4
python -m eole.bin.main predict -c recipes/qwen38/predict.yaml
EOLE_TORCH_COMPILE=1 EOLE_COMPILE_MODE=0 \
  python -m eole.bin.main serve -c recipes/qwen38/serve.yaml \
  --host 0.0.0.0 --port 5000
```

The server must bind to `0.0.0.0` inside Docker for forwarding. The example only
publishes to the host's loopback interface. From the host:

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/v1/models
```

Open `http://127.0.0.1:5000/` for the bundled workspace. See the
[UI recipes](../recipes/ui/README.md) for Chat, Translate, Documents (OCR),
Knowledge (RAG) and the combined showcase; mount/configure their models too.
See [Qwen3.8](../recipes/qwen38/README.md) for conversion and MTP,
[server](../recipes/server/README.md) for API examples, and
[installation](../README.md#installation) for optional `causal-conv1d`.
Training uses the same YAML interface:
`python -m eole.bin.main train -c /path/to/train.yaml`; mount datasets and outputs.
Stop and remove the container with `docker stop eole-dev` and `docker rm eole-dev`.

## Publish with GitHub Actions

After merging this Docker refresh, open **Actions → Build Docker Images → Run
workflow**. Select the updated workflow branch (normally `main`), then:

- `source_ref`: `main` for a development image, or a release tag that includes
  these Docker scripts. Older releases without these scripts are rejected.
- `torch_version`: `2.14.1`.
- `cuda_version`: `13.2.1`.

The workflow checks out that source, builds/pushes its matching base, then
builds/pushes Eole. A release checkout gets its exact release tag; a non-release
checkout gets `dev-<12-character-commit>`. OCI labels record the source revision.
No unrelated release number is assigned to current main. Publishing requires
GHCR package write permission; the workflow uses its repository `GITHUB_TOKEN`.
**Build Base Docker Image** publishes just the base when needed.

Local publishing is also explicit:

```bash
docker login ghcr.io
docker/build_base.sh 2.14.1 13.2.1 --push
docker/build.sh "dev-$(git rev-parse --short=12 HEAD)" 2.14.1 13.2.1 --push
```

`build.sh` verifies release tags against HEAD and refuses publishing tracked
uncommitted changes. Use a clean checkout, since untracked source files may also
be included. Builds retain Docker layer caching; add `--no-cache` to a manual
`docker build` when deliberately rebuilding all layers. GitHub-hosted builds
verify packaging and CLI/dependencies, but GPU inference must be tested on GPU
hardware before claiming a new combination is supported.
