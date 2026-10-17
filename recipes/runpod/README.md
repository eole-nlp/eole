# Eole on RunPod

Run the Eole workspace in your browser on a rented GPU. The deployment prepares
**Qwen Chat, Hy-MT2 translation and DeepSeek-OCR documents** automatically.
You do not need to install Eole, download models or run terminal commands.

**Deploy button: not published yet.** The private deployment has been tested
with INT4 Chat. The expanded translation/OCR and BF16 setup still needs cloud
validation before we publish its image and public RunPod template. Once that is
done, this page will provide a deploy button that opens the preconfigured template.

## Get started

1. **Prepare your RunPod account.** Sign in and add credit for GPU rental.
   In **Secrets**, create a Secret named `eole_password` with a unique password
   of at least 16 characters. Keep the value: you will use it to sign in to Eole.
   This setup is only needed once.
2. **Open the Eole deploy button/template link** when published. Choose an
   **RTX 5090 (32 GB)** for the default INT4 model, or an **RTX PRO 6000 (96 GB)**
   for more memory. Keep the template's disk, port and environment defaults.
   Check that the deployment includes its **250 GB volume at `/workspace`**,
   review the displayed price, then click **Deploy**.
3. **Open your workspace.** In the Pod's **Connect** tab, open the HTTP service
   on **port 5000**. Sign in with username **`eole`** and the password **value**
   you saved in `eole_password` (not your RunPod account password).
4. **Wait for preparation, then start using Eole.** The progress page opens the
   workspace automatically once downloads and conversion finish. First use of
   each model also loads it onto the GPU and compiles inference; the UI reports
   warmup separately. This can take several minutes.

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

In RunPod, choose **Stop** to end GPU billing and keep the volume for another
session. **Storage remains billable while stopped.** Restarting reuses completed
model downloads/conversions and compatible caches, though inference may warm up again.

Choose **Terminate** only when you no longer need the Pod: its Pod volume is
deleted. Download anything you want to keep first. Check the price RunPod shows
for both compute and storage.

## Need help?

- **Sign-in fails:** use lowercase `eole` and the value of your `eole_password`
  Secret. After changing that Secret, reset the Pod so it receives the new value.
- **Preparation stops:** open the Pod's **Logs → Container** for the error.
  Check GPU memory and available volume space before restarting.
- **First response is slow:** wait for model loading and compilation. Measure
  speed after warmup with the built-in Benchmark workspace.

Maintainers: [publish the image and deploy button](MAINTAINERS.md). That guide also
contains advanced settings, API access and local Docker testing.
