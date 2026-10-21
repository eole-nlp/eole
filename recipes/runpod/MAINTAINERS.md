# RunPod publishing and technical reference

For deploying an existing template, follow the [user quick start](README.md).
This guide is for maintainers publishing the images and deploy button, or users
customizing the deployment. Ordinary users do not need Docker commands or an API key.

## Publish an update with bundled CUDA ops

No new PyPI version is required. Existing images are immutable for this workflow:
publish new development/revision tags rather than overwriting `0.6.4` or
`0.6.4-r1`.

1. Merge both the precompiled-ops and pre-converted-model changes. Build the full
   Eole parent with **Actions → Build Docker Images** from `main`, using the merged
   commit, desired PyTorch/CUDA versions and architecture list. A non-release ref
   publishes a `dev-<commit>` image. If you already built a compatible parent from
   merged #473, reuse it: this model-download change only needs a new RunPod wrapper.
   The wrapper also supplies a matching Torchvision wheel for RAG when the parent
   lacks it, constrained to the parent's exact PyTorch version and CUDA wheel index.
2. Run **Actions → Build RunPod Image** from `main` with:
   - `source_ref`: the merged commit containing this model-download change.
   - `eole_image`: the full Eole image tag/digest published in step 1.
   - `deploy_tag`: **`0.6.4-r2`** for this update, or empty for testing only.
   The previous 0.6.4 parent lacks bundled ops and is intentionally rejected.
3. Verify the published image is public and test a disposable Pod. Build-time
   import checks do not replace GPU inference validation.
4. Once validated, update existing public RunPod template **`e9e19ozt9c`** to
   **`ghcr.io/eole-nlp/eole-runpod:0.6.4-r2`**. The repository's `template.json`
   already records this image. Keep the launcher,
   Secret reference, ports and storage settings. Merging repository changes or
   publishing an image does not update the live template automatically.
5. Open the README's deploy button and verify the selected image. The template
   ID and deploy links stay unchanged; existing Pods keep their existing images.

The CUDA base may be reused if its Python/PyTorch/toolkit environment is unchanged;
**Build Docker Images** also builds/publishes the matching base. The full Eole
image must be rebuilt to include the extension and updated source. Do not move an
existing release tag just to publish a Docker correction.

For development builds, leave `deploy_tag` empty to publish only `dev-<commit>`.
The named deployment tag is explicit, independent of the Python version and
source ref. Building/publishing an image does not rent a Pod. See the
[Docker guide](../../docs/docker.md) for base-image builds when those are needed.

The template uses 60 GB container disk, a 250 GB volume at `/workspace`, port
`5000/http`, the image's Python launcher and `EOLE_QWEN_VARIANT=int4`. Its password
is the deploying account's `eole_password` Secret reference. Do not use temporary
pinned-source bootstrap commands from private validation Pods. Pod volumes survive
Stop but not Terminate; network volumes have a separate lifecycle.

## Storage provisioning and the post-0.6.4 guard

RunPod's deploy form displays the template's `volumeInGb=250` and
`volumeMountPath=/workspace` as a recommendation. It allows deployment with no
volume attached, even though the template contains both fields. Changing those
fields alone cannot force a mount. The user must choose **Add volume → Volume
disk → 250 GB → Add volume disk** and verify the attached-volume summary.
See [RunPod's storage configuration step](https://docs.runpod.io/get-started#configure-storage).

The launcher validates Linux mount information before compiling kernels or
fetching models. It rejects the root filesystem and volatile mounts, including
cache paths pointed outside persistent storage. Space estimates account for
one copy of each pinned pre-converted checkpoint plus 8 GiB for download/compiler
headroom, with completed checkpoints excluded. Downloads go directly into resumable
staging directories; there is no second HF weight cache or conversion output. Filesystem space reports on shared storage may not
reflect per-volume quotas; verify the provisioned capacity in RunPod as well.
Storage errors are shown to the authenticated owner on the preparation page.

The guard is included in the **0.6.4-r1 RunPod wrapper**, using the existing
0.6.4 Eole base image. Follow the publishing steps above to activate it on the
public template.

## Optional: validate future launcher PRs without publishing images

Keep the existing released image and use a private test Pod with a fresh 250 GB
volume. Pin the PR commit locally, then use the Pod web terminal (or SSH when
configured) to place its changed launcher files in `/workspace/eole-pr/`.
Override only that Pod's container start command, for example:

```json
{"entrypoint":["/bin/bash","-lc"],"cmd":["set -e; cp /workspace/eole-pr/launch.py /workspace/eole-pr/preparing.html /opt/eole-runpod/; exec python /opt/eole-runpod/launch.py"]}
```

Stage the files before applying the override; editing the Pod restarts its
container. Include any other changed launcher dependencies when needed. This
tests launcher changes against the released Eole runtime, not changes to Eole's
installed Python code or CUDA sources. Verify file hashes, authentication,
fresh downloads/checksum verification, Chat/Translate/PDF OCR, model unloading and a cached
restart before publishing a final image. Keep this override out of the public
template. No PyPI release, version bump or image publication is needed for this
validation.

After validation, save the results and terminate disposable test Pods to remove
their attached Volume disks and end storage charges. Stop retains billable
storage and should be used only when another session is planned. Agree on
retention before testing; separately created network volumes need separate cleanup.

If HF metadata works but Xet transfers stall on a provider host, try
`HF_HUB_DISABLE_XET=1` in the test Pod environment to use ordinary HTTPS downloads.
Record this override with the results rather than treating it as the default.

## Local image build

With a matching Eole image containing bundled CUDA ops already built:

```bash
export EOLE_IMAGE=ghcr.io/eole-nlp/eole:NEW_FULL_IMAGE_TAG
docker build -f recipes/runpod/Dockerfile \
  --build-arg "EOLE_IMAGE=$EOLE_IMAGE" \
  --build-arg "VCS_REF=$(git rev-parse HEAD)" \
  -t ghcr.io/YOUR_ACCOUNT/eole-runpod:NEW_DEPLOY_TAG .
docker push ghcr.io/YOUR_ACCOUNT/eole-runpod:NEW_DEPLOY_TAG
```

## Register another template (optional)

The existing public template should normally be edited in the console. The
helper below creates a new template each time; it does not update `e9e19ozt9c`
and never rents a Pod. For a separate private test template:

```bash
export RUNPOD_API_KEY=YOUR_PRIVATE_RUNPOD_API_KEY
python recipes/runpod/publish_template.py \
  --image ghcr.io/eole-nlp/eole-runpod:dev-YOUR_COMMIT --dry-run
python recipes/runpod/publish_template.py \
  --image ghcr.io/eole-nlp/eole-runpod:dev-YOUR_COMMIT
```

Add `--public` only when publishing a separate tested template. Never include
actual passwords, API keys or HF tokens in template metadata; keep the Secret
placeholder so each deploying account supplies its own credential.

## Pre-converted checkpoint catalog

`models.json` records public Eole repositories, immutable commit pins, expected
file sizes and upstream attribution for INT4 Qwen, BF16 Qwen, Hy-MT2 and DeepSeek-OCR.
All four packages passed a fresh private-Hub download, complete SHA256 comparison
and offline inference before publication. INT4 (with MTP), translation and OCR
were tested on RTX 5090; the original BF16 checkpoint was tested on CPU using
Eole's PyTorch fallback. This does not constitute BF16 GPU validation of r2.

To update a model, validate the new Eole package first, then change its catalog
commit and size together and rebuild the RunPod wrapper. The Eole parent image
can be reused if its source/runtime and ops remain compatible. No PyPI release
is needed. The first r2 publication must include **both merged PRs**.
Existing r1 Pods/templates do not update themselves.
The public template must be changed only after the r2 image is available and
validated. Never advertise an unpublished image as currently deployed.

## Preparation and persistence

The entrypoint starts the gateway immediately, then prepares the backend:

- Fail closed if `EOLE_PASSWORD` is missing, too short or an unresolved Secret.
- Check GPU access/VRAM and probe the bundled multi-architecture Eole extension
  with RMSNorm on every visible GPU. Normal deployments do not compile Eole ops.
  If the binary is absent, try the matching volume cache. If the probe fails,
  compile for the visible GPUs and probe again. Caches are keyed by source,
  Python, PyTorch, compiler and visible GPU architectures. Fallback compilation
  uses `MAX_JOBS=2` by default. Triton/Inductor compilation remains separate.
- Download the selected Qwen variant(s), Hy-MT2-7B and DeepSeek-OCR from public
  `eole-nlp/*-eole` repositories, pinned by full commit SHA in `models.json`.
  No HF token is required. Original HF weights are not downloaded or converted.
- Download directly into a `.partial` model directory; interrupted transfers resume
  in place. Verify every manifest file's size and SHA256, plus the shard index,
  before writing the completion marker and atomically renaming the directory.
  Published configs use `${MODEL_PATH}` for portable tokenizer paths.
  Cached complete checkpoints skip downloading and hashing on restart. Older
  source-conversion caches are preserved but not selected by the new cache identity.
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
avoids another download and reuses matching kernels. One startup
writer owns this directory; use different `EOLE_DEPLOY_DIR` values if multiple
Pods share a network volume. Container-local `/eole` can be recreated safely.

| Variable | Default / purpose |
|---|---|
| `EOLE_USERNAME` | `eole`; browser sign-in / Basic-auth username |
| `EOLE_PASSWORD` | Required private Secret, at least 16 characters; also API Bearer token |
| `EOLE_DEPLOY_DIR` | `/workspace/eole`; persistent deployment directory |
| `EOLE_QWEN_VARIANT` | `int4`, `bf16`, or `both`; selects which Qwen checkpoints to prepare |
| `EOLE_BF16_MODEL_PATH` / `EOLE_BF16_MODEL_REVISION` | Existing BF16 checkpoint / HF revision when selecting `both` |
| `EOLE_TRANSLATION_MODEL_PATH` / `EOLE_TRANSLATION_MODEL_REVISION` | Existing Hy-MT2 checkpoint / full Hub commit SHA |
| `EOLE_OCR_MODEL_PATH` / `EOLE_OCR_MODEL_REVISION` | Existing DeepSeek-OCR checkpoint / full Hub commit SHA |
| `EOLE_MODEL_REVISION` | Full Hub commit SHA; defaults to the catalog pin for the selected Qwen variant |
| `EOLE_MODEL_PATH` | Optional converted default Qwen (INT4, or BF16 when selected alone); skips its download/verification |
| `HF_TOKEN` | Not required for the public showcase checkpoints; optional for Hub authentication |
| `EOLE_COOKIE_SECURE` | `1`; set `0` only for a local HTTP-only test |
| `MAX_JOBS` | `2`; CUDA compilation parallelism |
| `EOLE_TORCH_COMPILE` / `EOLE_COMPILE_MODE` | `1` / `0` |

An existing model must have valid container-side tokenizer paths. Do not copy a
config containing inaccessible paths from your workstation. Model revision overrides must be full commit SHAs in the corresponding
pre-converted repository. Changing a pin creates a distinct deployment cache;
branch names such as `main` are rejected to avoid silently reusing stale weights. Compiler caches are accelerators, not a
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

## Pre-converted workflow validation (21 October 2026)

Local Docker validation used the multi-architecture parent built for #473,
PyTorch 2.14.1 / CUDA 13.2 and a single RTX 5090. The new wrapper built without a
GPU, downloaded DeepSeek-OCR anonymously from its pinned public commit, verified
all files and brought up the authenticated workspace. INT4 and translation reused
portable packages mounted read-only. Compiled mode 0 exercised MTP Chat → Hy-MT2
translation → OCR with layout boxes → MTP Chat, with only Qwen loaded at the end.
The short OCR check recovered the synthetic numeric marker; it is not an OCR
quality benchmark. A cached restart reused the completed OCR checkpoint.

The additional RAG test exposed the missing Torchvision processor dependency.
The Docker helper now installs the compatible vision wheel from the selected
CUDA index while constraining PyTorch to its existing exact version, preserving
the extension ABI. The `rag` extra declares the image processor dependency too.
Build-time checks import both Torchvision and Eole ops. GPU EmbeddingGemma
indexing, source retrieval and a compiled MTP answer also passed in the corrected
wrapper, returning the synthetic document's requested code.

Sixty focused CPU tests, Black 24.10.0, Flake8 and a strict Sphinx build passed.
The four public catalog pins and sizes were checked anonymously against Hub
manifests. Every published model had already passed fresh-download checksum and
offline inference validation before this launcher change; BF16's publication
inference test used CPU. No new paid RunPod session or GHCR image was published
for this validation. Test the published r2 image through RunPod's HTTPS proxy
before updating the public template.

## Earlier cloud validation

Private RunPod validation on 18 October 2026 used one RTX PRO 6000 (96 GB),
the published Eole 0.6.3 / PyTorch 2.14.1 / CUDA 13.2 image, compiled mode 0,
and a 250 GB volume. The original stopped Pod's GPU was unavailable, so RunPod
migrated its data to another RTX PRO 6000. The test reused INT4 weights and
freshly downloaded/converted original BF16 Qwen, Hy-MT2-7B and DeepSeek-OCR.

| Check | Result |
|---|---|
| Browser-session sign-in through the public HTTPS proxy | Passed |
| INT4 Qwen Chat with compiled MTP | Returned the requested text |
| Hy-MT2 English → French translation | Returned “Bonjour, l’espace de travail de traduction est prêt.” |
| PDF import, rasterization and compiled DeepSeek-OCR | Recovered the synthetic document's text |
| Original BF16 Qwen Chat with compiled MTP | Returned the requested text on the 96 GB GPU |
| Switching INT4 → translation → OCR → BF16 → INT4 | Only the selected inference model remained loaded after each request |
| Restart with corrected converter | Reused valid checkpoints; automatically reconverted OCR and passed PDF OCR again |
| Local deployment-image build from the published base | Passed, including the packaged converter fix |

Fresh OCR conversion exposed a zero-width vision-head configuration. The fix
clears the generic zero head width for DeepSeek's nested CLIP configuration,
allowing Eole to derive 64 from hidden size 1024 / 16 heads. All 2,806 OCR tensors
matched their source; the failure was in configuration, not weights. Release 0.6.4 includes the converter correction in the Eole base image.
The OCR conversion cache is versioned to avoid reusing affected checkpoints.

Fifty targeted CPU tests passed, with Black and Flake8 checks. Tests cover
conversion staging/caches, workspace routing, authentication, proxy streaming,
process cleanup, runtime cache invalidation and the OCR head-width regression.
The initial private test on 17 October also validated fresh INT4 preparation and
GPU extension compilation. Local RTX 5090 Docker tests exercised compiled MTP
Chat/SSE and cache reuse.

These are functional compatibility tests with short synthetic inputs, including
first-use compilation; their request durations are not throughput benchmarks.
Knowledge/RAG indexing and other GPU families were not part of this validation.
The earlier storage correction was published as `eole-runpod:0.6.4-r1`. The
deploy button keeps template ID `e9e19ozt9c`; the prebuilt-kernel update targets
`0.6.4-r2` as described above. The fresh RTX 5090
validation of #470 also passed all three workspaces and a cached restart using
the existing image with the PR launcher injected. Its test Pod and attached volume
were terminated after validation; no test storage is retained.

### Publishing images with bundled CUDA ops

First run **Build Docker Images** on a ref containing the precompiled-ops change.
Its default targets are `8.0 8.6 8.9 9.0 10.0 10.3 12.0`, compiled with the
selected PyTorch/CUDA environment on a GPU-less builder. Adjust `cuda_arch_list`
if selecting an older toolkit. See [Docker documentation](../../docs/docker.md).
Then set **Build RunPod Image → eole_image** to the resulting full image tag or
digest. Older parent images without importable ops now fail that build explicitly.
Publish `0.6.4-r2` and update the public template to it after validation.
The repository template already points to r2; do not overwrite older tags. No PyPI release is required for this
Docker-only update. Test real inference on each claimed GPU family before
advertising it as validated; the build only cross-compiles those targets.
