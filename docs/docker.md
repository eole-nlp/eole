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

## Start a GPU container

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
python -c 'import torch; from eole import _ops; print(_ops.__file__)'
python -c 'from flash_attn import flash_attn_func, flash_attn_with_kvcache'
python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule'
```

New full images compile Eole ops during the image build, against that image's
Python/PyTorch/CUDA environment. No physical GPU is required on the builder.
The default `TORCH_CUDA_ARCH_LIST` is:

```text
8.0 8.6 8.9 9.0 10.0 10.3 12.0
```

This includes Ampere (A100, RTX 3090), Ada (RTX 4090, L40S), Hopper
(H100/H200), and Blackwell (B200/GB200, B300/GB300, RTX 5090/RTX PRO 6000).
These are compilation targets, not a claim of GPU testing on every family.
The selected CUDA toolkit must support every requested architecture; older
combinations may require a smaller list. Jetson and DGX Spark ARM64 deployments
are outside these x86_64 images' scope. Native code is built for each listed
architecture; future unlisted architectures may need rebuilding.

`docker/Dockerfile` accepts `TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` build arguments.
For example, `--build-arg TORCH_CUDA_ARCH_LIST="8.9 12.0" --build-arg MAX_JOBS=2`.
The local `docker/build.sh` and publishing workflow expose the same controls.
`MAX_JOBS` defaults to 2 to limit compiler memory use. Multi-architecture builds
cost more CI time and image space but avoid doing that work on a rented GPU.
The image label `io.eole.cuda.architectures` records the requested targets.
Changing PyTorch/CUDA or kernel sources requires a matching rebuilt extension.

For older images or a changed environment, `eole-build-kernels` remains a manual
fallback inside a running GPU container. With no explicit architecture list it
targets all visible GPUs and forcibly recompiles the extension. It only builds
the extension; the image already contains Eole's dependencies. It operates on
`/eole`, not a mounted native host checkout. To cross-compile without a GPU, pass
`TORCH_CUDA_ARCH_LIST` explicitly. Never reuse a host `.so` blindly across images.

Triton JIT, `torch.compile` and graph warmup are separate from this extension and
still run as needed. GPU-less image builds check import/ABI compatibility; real
GPU execution must be tested separately.

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

The Qwen server recipes use `preload: true` and `warmup: true`: preparation finishes
before the HTTP server accepts requests. With single-GPU compile mode 0 and MTP,
the first startup compiles only the full verifier chunk and retains it in memory;
later startups can load the locally saved program. No compiled verifier is downloaded.
Mount a persistent cache, for example `-v "$PWD/eole-cache:/root/.cache/eole"`, when
creating the container. Optionally set `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`
and `CUDA_CACHE_PATH` to subdirectories of that mount to retain other kernel caches
too. `EOLE_VERIFIER_CACHE_DIR` overrides the verifier directory. Cache compatibility
is checked against the runtime, GPU, source and model layout; failures fall back
to ordinary compilation. Input shapes not covered by preparation can still warm
additional kernels. These features require an image built from the updated source.

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
- `cuda_arch_list`: `8.0 8.6 8.9 9.0 10.0 10.3 12.0`.
- `max_jobs`: `2`.

The workflow checks out that source, builds/pushes its matching base, then
builds/pushes Eole. A release checkout gets its exact release tag; a non-release
checkout gets `dev-<12-character-commit>`. OCI labels record the source revision.
No unrelated release number is assigned to current main. Publishing requires
GHCR package write permission; the workflow uses its repository `GITHUB_TOKEN`.
**Build Base Docker Image** publishes just the base when needed.

For the **0.6.4 release image**, after the version-bump PR is merged and the
`0.6.4` tag is created on the release commit, select `source_ref=0.6.4` (keep the
workflow branch on `main`). The workflow then publishes:

```text
ghcr.io/eole-nlp/eole:0.6.4-torch2.14.1-ubuntu24.04-cuda13.2.1
```

The release tag must include the refreshed Docker tooling. Publishing a GitHub
release triggers the Python package/docs workflow; Docker publication still
requires the separate manual workflow above.

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

### RunPod deployment images

Build/publish the full Eole image from a commit containing the precompiled-ops
change first. A compatible parent already built from #473 can be reused; the
pre-converted-model changes only require rebuilding the RunPod wrapper from
the newly merged source commit. Both image builds install a matching Torchvision
wheel for the RAG processor, constrained to the selected PyTorch/CUDA runtime.
Then run **Build RunPod Image** with `eole_image` set to that newly published tag
(or digest), and `deploy_tag=0.6.4-r2`. The RunPod build rejects parent images that
lack importable Eole ops, rather than silently publishing an image that compiles
on every first deployment. Existing 0.6.4/0.6.4-r1 images are not changed by this PR.

The repository template targets `0.6.4-r2`. Update the public template's
`imageName` to that tag after publishing and validation, using
[the template publishing instructions](../recipes/runpod/MAINTAINERS.md).
Changing Git source alone does not update an existing template or running Pod.
The r2 launcher downloads pre-converted public `eole-nlp` checkpoints pinned in
`recipes/runpod/models.json`, then verifies file SHA256 checksums. It does not
download original HF weights or run model conversion on the Pod.
On startup the launcher probes the bundled extension on each visible GPU, then
uses source compilation only if the binary is absent or incompatible. A matching
volume cache is used when no bundled binary exists. The fallback result is cached.

The full-image workflow also uploads a downloadable `eole-cuda-ops-*` Actions
artifact containing the `.so` and a SHA256/compatibility manifest (source commit,
Python ABI, PyTorch, CUDA, CPU architecture, glibc, C++ ABI and GPU targets).
This is an advanced exact-environment artifact, not a universal wheel or an
automatic installer. Normal Docker/RunPod users get the binary through the image
layers. Installing an exported `.so` elsewhere requires matching its manifest;
rebuild from source when any compatibility requirement is uncertain.
