---
sidebar_position: 5
description: How to configure stdout, TensorBoard, and Trackio reporting during training.
---

# Experiment Reporting

EOLE supports multiple reporting backends during training:

- stdout (always enabled)
- TensorBoard (optional)
- Trackio (optional)

## Stdout Reporting

Stdout reporting writes periodic training metrics directly to the process logs and is always enabled.

Use `report_every` to control the step interval for these logs.

```yaml
report_every: 100
```

## TensorBoard Reporting

[`TensorBoard`](https://www.tensorflow.org/tensorboard) is a local dashboard for visualizing scalar metrics over time.

Enable TensorBoard in your train config with:

```yaml
tensorboard: true
tensorboard_log_dir: runs/eole
```

When enabled, EOLE creates a dated run directory under `tensorboard_log_dir` and logs training/validation metrics.

To visualize logs:

```bash
tensorboard --logdir runs/eole
```

## Trackio Reporting

[`Trackio`](https://huggingface.co/docs/trackio/index) is an experiment tracking backend for local-first logging and optional remote dashboards.

Trackio is an optional dependency:

```bash
pip install -e ".[trackio]"
```

The extra installs `trackio` (>=0.35,<0.38), `psutil` (CPU/system metrics on all platforms, including Apple Silicon GPU stats), and `nvidia-ml-py` (NVIDIA GPU metrics; skipped on macOS, where CUDA is not available).

Enable Trackio in your train config:

```yaml
trackio: true
trackio_project: eole
trackio_run_name: my-run-name  # optional
trackio_group: experiments-v1  # optional
trackio_space_id: username/space_id  # optional
trackio_bucket_id: my-bucket-id  # optional

trackio_auto_log_cpu: null  # null = auto-detect
trackio_auto_log_gpu: null  # null = auto-detect
trackio_system_log_interval: 10.0
trackio_log_config_artifact: true
```

All Trackio options:

| Option | Default | Description |
| --- | --- | --- |
| `trackio` | `false` | Enable Trackio reporting during training. |
| `trackio_project` | `eole` | Trackio project name. |
| `trackio_run_name` | auto-generated | Run name. |
| `trackio_group` | unset | Organizes related runs together in the dashboard. |
| `trackio_space_id` | unset | Hugging Face Trackio Space target for a remote dashboard. |
| `trackio_bucket_id` | unset | Hugging Face bucket for metric persistence. |
| `trackio_auto_log_cpu` | `null` (auto) | Log CPU/RAM/disk/network/sensor metrics (requires `psutil`). Use `true`/`false` to force. |
| `trackio_auto_log_gpu` | `null` (auto) | Log GPU metrics (`nvidia-ml-py` on NVIDIA hardware, `psutil` on Apple Silicon). Use `true`/`false` to force. |
| `trackio_system_log_interval` | `10.0` | Seconds between automatic system (CPU/GPU) metric samples. |
| `trackio_log_config_artifact` | `true` | Upload the run's configuration as file artifacts (see below). |

EOLE also logs run metadata to Trackio's run config: the model and training sections, plus `eole_version` and the `git_commit` the run was launched from.

### Config Artifacts

With `trackio_log_config_artifact: true`, each run gets two artifacts:

- The original config artifact uses the original basename, sanitized for Trackio artifact safety (characters outside `A-Z a-z 0-9 . _ -` are replaced with `-`). For example `wmt17-mini.yml` → `wmt17-mini.yml`, `my run (v2).yaml` → `my-run--v2-.yaml`. It is the exact file you launched with, uploaded byte-for-byte.
- The effective config artifact is named `<stem>-effective<suffix>` (e.g. `wmt17-mini-effective.yml`) and contains the complete runtime config after validation and default-filling. All defaults are included, and non-YAML-native values are stringified (e.g. `compute_dtype: torch.bfloat16`). For non-YAML source configs (e.g. `config.json`), the effective artifact uses the `.yaml` extension since its content is YAML.

Runs launched from CLI flags only (no `--config` file) upload just `config-effective.yaml`.

Artifacts are versioned per project: repeated runs in the same project land as `v0`, `v1`, ..., each linked to its own run via lineage, so earlier configs are never overwritten.

At runtime, the following failure behavior happens:
- If Trackio is not installed, EOLE logs a warning and continues training.
- If Trackio initialization fails, EOLE logs a warning and continues training.
- If Trackio raises during runtime logging, EOLE logs a warning once and disables Trackio for the rest of the run.
- If config artifact upload fails, EOLE logs a warning and metric logging continues.

## Distributed behavior

In distributed training, TensorBoard and Trackio are initialized on rank 0 (`gpu_rank <= 0`) only, to avoid duplicate logging.

## Config snippet example

```yaml
report_every: 100

tensorboard: true
tensorboard_log_dir: runs/eole

trackio: true
trackio_project: eole
trackio_run_name: exp-001
trackio_group: experiments-v1
```

## Extending with a Custom Report Backend

For advanced integrations, implement your own manager by subclassing `ReportMgrBase` from `eole.utils.report_manager` and then compose it with `CompositeReportMgr`.

At minimum, implement:

- `_report_training(...)` for step-level periodic logging
- `_report_step(...)` for end-of-step train/valid summaries
- `close()` if your backend owns external resources

If you instantiate a manager directly, call `start()` before the first `report_training(...)` call so `start_time` is initialized.

Example sketch:

```python
from eole.utils.report_manager import CompositeReportMgr, ReportMgrBase, StdoutReportMgr


class MyBackendReportMgr(ReportMgrBase):
    def _report_training(self, step, num_steps, learning_rate, patience, report_stats):
        # send training metrics to your backend
        pass

    def _report_step(self, lr, patience, step, valid_stats=None, train_stats=None):
        # send step summary to your backend
        pass


report_mgr = CompositeReportMgr(
    report_every=100,
    managers=[
        StdoutReportMgr(report_every=100),
        MyBackendReportMgr(report_every=100),
    ],
)
```