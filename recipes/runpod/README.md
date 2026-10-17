# Deploy Eole Chat on RunPod

This recipe packages an automatically prepared Qwen Chat Pod. Users open a shared
RunPod template, select a GPU, confirm its cost, and open the workspace through
RunPod's HTTPS link. No terminal commands are needed **after the template is
published**. First-time users still need a RunPod account, payment setup, and a
private password Secret. Downloads, conversion and compilation take time.

The recipe is ready for local testing and template registration. **There is no
live public template ID yet.** A maintainer must publish the image, test a private
Pod, and register the public template before adding a working deploy button.
Local Docker validation does not establish RunPod hardware/proxy compatibility.

## User experience

1. Create a [RunPod Secret](https://docs.runpod.io/pods/templates/manage-templates#using-secrets-in-templates)
   named `eole_password`, with a unique password of at least 16 characters. This
   is your own account's secret; the public template never contains a shared password.
2. Open the maintainer's public template link. Choose **one NVIDIA GPU with at
   least 32 GB advertised VRAM**, on an **x86-64 host** with a driver supporting
   CUDA 13.2. RTX 5090 is locally tested; H100/H200 are expected compatible but
   not hardware-validated. Grace/ARM systems need another image. Use at least
   32 GB host RAM; 64 GB is preferable for conversion and kernel compilation.
3. Confirm GPU availability, disk settings and rental cost, then deploy.
4. Open **Connect → HTTP service on port 5000**. Sign in as `eole` with the
   password you saved. A preparation page reports GPU checks, kernel preparation,
   download, conversion and model loading. It automatically opens the workspace
   when ready. The first inference request additionally compiles decoding/MTP;
   the existing UI reports that warmup separately.
5. Use Chat with text/images and the built-in benchmark. Qwen is the provisioned
   model; the dedicated translation/OCR models are not downloaded by this recipe.
6. Stop/terminate the Pod when done. Compute and storage have separate charges;
   check RunPod's current billing/storage rules. Keep needed files on your volume.

RunPod exposes the UI at `https://<pod-id>-5000.proxy.runpod.net/` and APIs at
that address plus `/v1`. The URL is reachable from other computers. **HTTPS
provides encryption, not authorization**: this recipe requires authentication on
the workspace, preparation status, static assets, health, APIs and uploads.
Only the generic sign-in page and login submission are public.
The browser opens a normal sign-in page without a native HTTP-auth prompt.
Only expose `5000/http`; the Eole backend is bound privately to `127.0.0.1:5001`.
Never expose backend port 5001. The gateway does not add an SSH service.

See RunPod's official [template configuration](https://docs.runpod.io/pods/templates/manage-templates),
[port exposure](https://docs.runpod.io/pods/configuration/expose-ports) and
[storage](https://docs.runpod.io/pods/storage/types) documentation.

## Maintainer: build the deployment image

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

## Maintainer: register and test the template

Inspect `template.json`. Its defaults are 60 GB container disk, 100 GB persistent
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
- Download `RedHatAI/Qwen3.8-27B-INT4` into the volume's HF cache. Only that
  model/layout is configured here; arbitrary model IDs are not interchangeable.
- Convert into a staging directory, relocate output tokenizer paths, and rename
  only after completion. Interrupted conversions are retried, not reused.
- Run the existing `recipes/ui/serve.yaml` settings privately: BF16, FlashAttention,
  packed projections, 28,672-token context, 4,096 output tokens, three MTP drafts
  and chunked prefill. FP32 normalization is unchanged.
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
| `EOLE_MODEL_REVISION` | `main`; set an HF commit SHA for reproducible model downloads |
| `EOLE_MODEL_PATH` | Optional existing converted Qwen directory; skips download/conversion |
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
cancellation and template defaults. The download/conversion orchestration is
mocked in these tests; a fresh full model download/conversion and an actual
RunPod deployment still need the private-Pod validation described above.
