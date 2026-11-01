# Eole on RunPod

Run the Eole workspace in your browser on a rented GPU. The deployment prepares
**Qwen Chat, Hy-MT2 translation and DeepSeek-OCR documents** automatically.
You do not need to install Eole, download models or run terminal commands.

[![Deploy on RunPod](https://img.shields.io/badge/RunPod-Deploy%20Eole-673AB7?style=for-the-badge)](https://console.runpod.io/deploy?template=e9e19ozt9c&type=gpu)

The button opens the public Eole template. **Before confirming deployment, attach
its required 250 GB model volume using the steps below.** RunPod only recommends
that volume; it does not attach it automatically. The 60 GB container disk alone
is not sufficient. Private cloud validation has passed INT4/BF16 Chat,
translation, PDF OCR and automatic model switching with an attached volume.

This recipe targets **`eole-runpod:0.6.5`**, with prebuilt Eole CUDA ops and
pre-converted models from the public [eole-nlp organization](https://huggingface.co/eole-nlp).
No HF token is required for these four checkpoints; no model conversion runs on the Pod.
Maintainers must first build a full Eole image containing those ops, publish the
RunPod wrapper, and update the existing public template before the button uses it; see the [publishing steps](MAINTAINERS.md#publish-065).

## Get started

1. **Prepare your RunPod account.** Sign in and add credit for GPU rental.
   In **Secrets**, create a Secret named `eole_password` with a unique password
   of at least 16 characters. Keep the value: you will use it to sign in to Eole.
   This setup is only needed once.
2. **Click Deploy Eole above.** Choose an
   **RTX 5090 (32 GB)** for the default INT4 model, or an **RTX PRO 6000 (96 GB)**
   for more memory. Keep the 60 GB container disk and other defaults.
   **Attach the model volume explicitly:** choose **Add volume → Volume disk →
   250 GB → Add volume disk**, mounted at **`/workspace`**. The template only
   recommends this volume; RunPod does not attach it automatically. Before
   clicking **Deploy**, verify **Attached volumes** lists the 250 GB Volume disk
   and the pricing breakdown includes its cost (**310 GB total disk**, including
   the 60 GB container disk). Do not proceed with “Nothing mounted at the
   template’s path”.
3. **Open your workspace.** In the Pod's **Connect** tab, open the HTTP service
   on **port 5000**. Sign in with username **`eole`** and the password **value**
   you saved in `eole_password` (not your RunPod account password).
4. **Wait for preparation, then start using Eole.** With images built from this
   revision, Qwen loads and warms inference before the workspace opens. Its first
   startup compiles the full MTP verification chunk and saves it locally; restarts
   reuse that verifier from the volume. Translation and OCR still load and warm
   on first use. First preparation can take several minutes.
5. **Clean up when finished.** For a disposable test, choose **Terminate** in
   RunPod to delete the Pod and its attached Volume disk and end their charges.
   Save any outputs you want first. Choose **Stop** only if you intend to resume:
   the retained volume continues to cost money even while the GPU is stopped.

The button preconfigures the software, models and server. RunPod still requires
an account, a private password, an available GPU and your approval of the rental
cost; it cannot skip those steps.

## What you can try

| Workspace | Prepared model / behavior |
|---|---|
| Chat | [Qwen3.8-27B INT4](https://huggingface.co/eole-nlp/Qwen3.8-27B-INT4-eole); text and image conversations |
| Translate | [Hy-MT2-7B](https://huggingface.co/eole-nlp/Hy-MT2-7B-eole); dedicated translation workflow |
| Documents (OCR) | [DeepSeek-OCR](https://huggingface.co/eole-nlp/DeepSeek-OCR-eole); upload PDFs/images and extract Markdown |
| Knowledge (RAG) | Qwen answers questions using your documents; the embedding model downloads on first use |
| Benchmark | Measure generation speed after warmup |

Images built from the updated launcher also prepare the pinned
`incoai/Qwen3.8-27B-DFlash2` drafter (about 3.85 GB), so Chat offers Off, MTP
and DFlash2 without a manual download. Config and weight SHA256 checksums are
verified before activation. INT4 and BF16 targets share one cached drafter on
the volume; downloads interrupted during preparation resume on restart. Storage
checks include this extra checkpoint. Native MTP remains the startup default.
Selecting DFlash2 or changing precision reloads and warms the engine on the next
request, so that request can take longer to produce its first token. This needs
a newly built full Eole image containing the workspace controls and a RunPod
wrapper built from the same updated source; existing published images do not
change automatically.

Switching workspaces automatically unloads the previous inference model and loads
the appropriate one. All prepared weights stay on disk. Loading and first-use
warmup take time, so a switch is not instantaneous.

### Startup compilation and restarts

The default Qwen Chat model is preloaded with `warmup: true`. The server drains a
short preparation request in its inference worker before `/health` becomes
available. The gateway shows **Load Qwen and prepare inference** until the model
is ready. The just-compiled verifier stays in memory and serves requests immediately.
No verifier artifact is downloaded or published.

`EOLE_VERIFIER_CACHE_DIR` defaults to `/workspace/eole/verifiers`. Artifacts are
generated for the local GPU, PyTorch/CUDA build, Eole code, model layout and cache
dimensions. Incompatible or corrupt entries are rebuilt; unsupported serialization
or unwritable caches fall back to ordinary full-chunk compilation. Keep the volume
to reuse the cache. Terminating the Pod and its volume deletes it. Short verification
tails remain eager. This covers single-GPU compile mode 0; mode 2, tensor parallel
and non-MTP paths retain their existing policy. New prompt/image shapes can still
need additional kernel warmup.

**Publishing:** these startup changes require a full Eole image and RunPod wrapper
built from this revision. Publishing the `0.6.5` image and updating the public template are separate steps;
merging this recipe leaves existing images and Pods unchanged. Update the template
using [MAINTAINERS.md](MAINTAINERS.md). The wrapper build rejects parent images
that lack the matching server/cache implementation.

## Optional: use original Qwen instead of INT4

For the default deployment, change nothing. To try the
[original BF16 model](https://huggingface.co/eole-nlp/Qwen3.8-27B-BF16-eole), change **`EOLE_QWEN_VARIANT`** under **Environment variables** before clicking Deploy:

| Value | Chat models available | GPU choice |
|---|---|---|
| `int4` — default | RedHat Qwen3.8-27B INT4 | 32 GB or larger |
| `bf16` | Original Qwen3.8-27B in BF16 | 80 GB minimum; 96 GB recommended |
| `both` | Both models, selectable in Chat | 80 GB minimum; 96 GB recommended |

Translation and OCR are included with every choice. Keep the **250 GB volume**.
The original BF16 model has a larger download. Pre-converted checkpoints avoid
conversion and its extra host RAM and disk requirements. For either option, use an **x86-64 host** and a driver
supporting **CUDA 13.2**; this image does not support Grace/ARM hosts.

## Open it from another computer

Use the same HTTPS address from **Connect → port 5000** on any computer:
`https://<pod-id>-5000.proxy.runpod.net/`. Sign in with your Eole credentials.
Keep the password private; it grants access to the workspace and its APIs.

## When you are finished

**For a completed demo or disposable test, save your outputs and choose
Terminate. Closing the browser or signing out does not stop billing.**

| RunPod action | Compute charges | Attached Volume disk |
|---|---|---|
| **Stop** | Stop | Retained and still billed; use when resuming later |
| **Terminate** | End | Permanently deleted, including models, caches and outputs; its storage charges end |

Stopping preserves completed downloads, verified models and compatible compiler
caches, so resuming avoids preparing everything again. Inference may still warm
up again. It is useful for another session, but unnecessary for a finished test.

For reference, the 250 GB test Volume disk was quoted at **$0.069/hour while
stopped (about $1.67/day)**. Prices can change: check RunPod's current price and
the final confirmation dialog. Termination cannot be undone. Separately created
network volumes have their own lifecycle and remain billable until deleted
separately; the quick start above uses an attached **Volume disk**.

## Need help?

- **“No persistent volume” / storage check fails:** the volume is missing, even if
  `/workspace` exists. In a new deployment choose **Add volume → Volume disk**,
  set **250 GB**, and check its mount is `/workspace`. A 60 GB container disk
  alone is insufficient. The launcher checks storage before kernels/downloads.
- **“No space left on device”:** check which disk is full and the mounted volume's
  quota. Checkpoints, partial downloads and compiler caches all need space.
  Increasing only the temporary container disk does not attach persistent storage.
  Stopping/resetting a volume-less Pod loses its container-local downloads.


- **Sign-in fails:** use lowercase `eole` and the value of your `eole_password`
  Secret. After changing that Secret, reset the Pod so it receives the new value.
- **Preparation stops:** open the Pod's **Logs → Container** for the error.
  Check GPU memory and available volume space before restarting.
- **First response is slow:** wait for model loading and compilation. Measure
  speed after warmup with the built-in Benchmark workspace.

Maintainers: [publish the image and deploy button](MAINTAINERS.md). That guide also
contains advanced settings, API access and local Docker testing.

New images built with bundled Eole CUDA ops check the binary at startup instead
of compiling it on the rented GPU. A source-build fallback handles incompatible
binaries. Model downloads, checksum verification and Triton/torch.compile warmup still take time.
Older published images retain their original startup behavior. Maintainers must
publish a new image and update the deployment template to enable this change;
see [publishing instructions](MAINTAINERS.md#publish-065).
