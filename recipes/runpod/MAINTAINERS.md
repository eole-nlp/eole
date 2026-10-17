# RunPod publishing and technical reference

For deploying an existing template, follow the [user quick start](README.md).
This guide is for maintainers publishing the deploy button, or users customizing
the deployment. Ordinary users do not need Docker commands or an API key.

## Build the deployment image

First publish the Eole 0.6.3 CUDA13.2 image using the [Docker guide](../../docs/docker.md).
Then run **Actions → Build RunPod Image** with a source ref containing this recipe.
Its `eole_image` input defaults to:

```text
ghcr.io/eole-nlp/eole:0.6.3-torch2.14.1-ubuntu24.04-cuda13.2.1
```

The workflow builds and publishes a separate immutable source tag such as
`ghcr.io/eole-nlp/eole-runpod:dev-<12-character-commit>`. It does not register a
template or rent a Pod. The published base image must already exist. Local build:

```bash
docker build -f recipes/runpod/Dockerfile \
  -t ghcr.io/YOUR_ACCOUNT/eole-runpod:YOUR_VERSION .
docker push ghcr.io/YOUR_ACCOUNT/eole-runpod:YOUR_VERSION
```

This thin image adds startup automation and an authenticated gateway. Normal
`eole serve` and the generic Docker image remain unchanged.

## Register and test the template

Inspect `template.json`. Its defaults are 60 GB container disk, 250 GB persistent
volume at `/workspace`, and HTTP port 5000. The volume must have room for both the
HF source weights and converted checkpoint. The template does not choose a GPU
or guarantee availability. Users may attach a network volume for persistence
beyond a Pod's lifetime; a Pod volume alone does not survive termination.

Register a **private** template first (requires Python `requests`):

```bash
export RUNPOD_API_KEY=YOUR_PRIVATE_RUNPOD_API_KEY
python recipes/runpod/publish_template.py \
  --image ghcr.io/eole-nlp/eole-runpod:dev-YOUR_COMMIT --dry-run
python recipes/runpod/publish_template.py \
  --image ghcr.io/eole-nlp/eole-runpod:dev-YOUR_COMMIT
```

The helper calls RunPod's template API only; **it never creates a billable Pod**.
It prints the template URL. Deploy a test Pod from that link in the console and
verify authentication, preparation, streamed Chat, image upload, restart/cache
reuse and shutdown. This test rents compute and requires your account/payment.

Once validated, register a public template by adding `--public` to the command.
The helper prints the real Markdown deploy link, ready to add to the main README
and website. It creates a new template on each invocation; retain the returned ID
and use the console to edit an existing listing rather than creating duplicates.
The Secret placeholder is resolved for the deploying account; never replace it
with your own password or include API keys/HF tokens in the public template.

## Preparation and persistence

The entrypoint starts the gateway immediately, then prepares the backend:

- Fail closed if `EOLE_PASSWORD` is missing, too short or an unresolved Secret.
- Check GPU access/VRAM. Compile Eole kernels if absent, then execute a small
  RMSNorm check. Kernel caches are keyed by source, Python, PyTorch, compiler and
  visible GPU architectures; mismatches trigger a rebuild. Compilation uses
  `MAX_JOBS=2` by default and targets all visible GPUs.
- Download the selected Qwen variant(s), `tencent/Hy-MT2-7B`, and
  `deepseek-ai/DeepSeek-OCR` into the volume's HF cache. These supported model
  IDs have specific workspace settings; arbitrary IDs are not interchangeable.
- Convert into a staging directory, relocate output tokenizer paths, and rename
  only after completion. Interrupted conversions are retried, not reused.
- Run the existing `recipes/ui/serve-showcase.yaml` settings privately: BF16, FlashAttention,
  packed projections, 28,672-token context, 4,096 output tokens, three MTP drafts
  and chunked prefill for Qwen. Translation/OCR retain their own recipe settings.
  `model_policy: single` and lazy loading keep only one inference model resident.
  FP32 normalization is unchanged.
- Stream requests through the authenticated gateway, with SSE heartbeats during
  long compilation/prefill waits so the public connection does not sit idle.
  Stop child process groups
  on shutdown. Preparation failures remain visible on the status page; detailed
  errors stay in private Pod logs.

`/workspace/eole` holds model weights, kernels and compiler caches. Reusing it
avoids another download/conversion and reuses matching kernels. One startup
writer owns this directory; use different `EOLE_DEPLOY_DIR` values if multiple
Pods share a network volume. Container-local `/eole` can be recreated safely.

| Variable | Default / purpose |
|---|---|
| `EOLE_USERNAME` | `eole`; browser sign-in / Basic-auth username |
| `EOLE_PASSWORD` | Required private Secret, at least 16 characters; also API Bearer token |
| `EOLE_DEPLOY_DIR` | `/workspace/eole`; persistent deployment directory |
| `EOLE_QWEN_VARIANT` | `int4`, `bf16`, or `both`; selects which Qwen checkpoints to prepare |
| `EOLE_BF16_MODEL_PATH` / `EOLE_BF16_MODEL_REVISION` | Existing BF16 checkpoint / HF revision when selecting `both` |
| `EOLE_TRANSLATION_MODEL_PATH` / `EOLE_TRANSLATION_MODEL_REVISION` | Existing Hy-MT2 checkpoint / revision (`main`) |
| `EOLE_OCR_MODEL_PATH` / `EOLE_OCR_MODEL_REVISION` | Existing DeepSeek-OCR checkpoint / revision (`main`) |
| `EOLE_MODEL_REVISION` | `main`; set an HF commit SHA for reproducible model downloads |
| `EOLE_MODEL_PATH` | Optional converted default Qwen (INT4, or BF16 when selected alone); skips its download/conversion |
| `HF_TOKEN` | Optional private HF credential via a RunPod Secret |
| `EOLE_COOKIE_SECURE` | `1`; set `0` only for a local HTTP-only test |
| `MAX_JOBS` | `2`; CUDA compilation parallelism |
| `EOLE_TORCH_COMPILE` / `EOLE_COMPILE_MODE` | `1` / `0` |

An existing model must have valid container-side tokenizer paths. Do not copy a
config containing inaccessible paths from your workstation. `main` revisions
reuse the saved conversion until `EOLE_MODEL_REVISION` changes; pin a commit SHA
when publishing a reproducible template. Compiler caches are accelerators, not a
promise that every restart skips warmup. Do not share an access password with
untrusted users: it grants the UI's model-management and inference capabilities.
Browser sign-in uses an HttpOnly, Secure, SameSite=Strict cookie, valid for 12 hours
and invalidated by a gateway restart. POST `/_deployment/logout` to end the session.
Basic and Bearer authentication remain available for API clients.

API clients can use the same secret without a browser login:

```bash
export EOLE_URL=https://YOUR_POD_ID-5000.proxy.runpod.net
curl -H "Authorization: Bearer $EOLE_PASSWORD" "$EOLE_URL/v1/models"
```

## Local end-to-end test

The current launcher prepares all three workspaces. To reuse local checkpoints,
set `EOLE_MODEL_PATH`, `EOLE_TRANSLATION_MODEL_PATH` and `EOLE_OCR_MODEL_PATH`
to their mounted container paths; otherwise missing models are downloaded.
For `both`, also set `EOLE_BF16_MODEL_PATH` if reusing the BF16 checkpoint.

Use a GPU Docker host, an existing converted model if available, and a private
environment file (never commit it):

```bash
# /absolute/private/runpod.env contains EOLE_PASSWORD=your-unique-long-password
# For this localhost HTTP test only, also set EOLE_COOKIE_SECURE=0.
# Add EOLE_MODEL_PATH=/models/Qwen3.8-27B-INT4 to reuse a converted checkpoint.
docker run --name eole-runpod-test --gpus all --shm-size=8g \
  --env-file /absolute/private/runpod.env \
  -v /absolute/persistent-directory:/workspace \
  -v /absolute/model-directory:/models:ro \
  -p 127.0.0.1:5000:5000 ghcr.io/YOUR_ACCOUNT/eole-runpod:YOUR_VERSION
```

Open `http://127.0.0.1:5000/` locally and log in. For real remote use, use RunPod's
HTTPS URL. Stop the container with `docker stop eole-runpod-test`; preparation
and backend subprocesses are terminated too. Keep the volume for the next run.

## Validation recorded for this PR

Local tests used a Docker image with PyTorch 2.14.1/CUDA 13.2, FlashAttention
2.8.3, FLA 0.5.2 and a prebuilt Eole extension, on an RTX 5090. The existing
converted RedHat checkpoint was mounted read-only. Authentication was checked
on the UI, docs, status and API; Basic/Bearer access and real compiled MTP SSE
Chat succeeded. A recreated container reused the persisted extension cache and
reached ready again. These short requests are compatibility tests, not speed
benchmarks. The cold first-request MTP warmup took about six minutes.

CPU tests cover authentication, proxy uploads/SSE cleanup and heartbeats,
conversion staging/reuse/path relocation, runtime cache invalidation, process
cancellation and template defaults. The multi-model download/conversion orchestration is mocked in these tests.
The private RunPod test subsequently validated a fresh INT4 download/conversion,
GPU extension build/cache, streamed compiled MTP Chat and browser sign-in behind
the RunPod proxy. It exposed and fixed an extension import-order problem,
a tokenizer location mismatch, and proxy Host rewriting in the origin check.
The Pod is stopped; no paid cloud test of BF16 or automatic translation/OCR
preparation has been run yet.
