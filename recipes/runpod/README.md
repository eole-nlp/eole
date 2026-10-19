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

The corrected deployment image is **`eole-runpod:0.6.4-r1`**, extending the
existing Eole **0.6.4** base. Maintainers must publish it and update the existing
template before the button uses it; see the [publishing steps](MAINTAINERS.md#publish-the-runpod-correction-as-064-r1).

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
4. **Wait for preparation, then start using Eole.** The progress page opens the
   workspace automatically once downloads and conversion finish. First use of
   each model also loads it onto the GPU and compiles inference; the UI reports
   warmup separately. This can take several minutes.
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
| Chat | Qwen3.8-27B INT4; text and image conversations |
| Translate | Hy-MT2-7B; dedicated translation workflow |
| Documents (OCR) | DeepSeek-OCR; upload PDFs/images and extract Markdown |
| Knowledge (RAG) | Qwen answers questions using your documents; the embedding model downloads on first use |
| Benchmark | Measure generation speed after warmup |

Switching workspaces automatically unloads the previous inference model and loads
the appropriate one. All prepared weights stay on disk. Loading and first-use
warmup take time, so a switch is not instantaneous.

## Optional: use original Qwen instead of INT4

For the default deployment, change nothing. To try the original unquantized model,
change **`EOLE_QWEN_VARIANT`** under **Environment variables** before clicking Deploy:

| Value | Chat models available | GPU choice |
|---|---|---|
| `int4` — default | RedHat Qwen3.8-27B INT4 | 32 GB or larger |
| `bf16` | Original Qwen3.8-27B in BF16 | 80 GB minimum; 96 GB recommended |
| `both` | Both models, selectable in Chat | 80 GB minimum; 96 GB recommended |

Translation and OCR are included with every choice. BF16 preparation also needs
at least **128 GB host RAM**; keep the **250 GB volume**. Its larger download and
conversion take longer. For either option, use an **x86-64 host** and a driver
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

Stopping preserves completed downloads, converted models and compatible compiler
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
  quota. Downloads, converted checkpoints and temporary output all need space.
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
