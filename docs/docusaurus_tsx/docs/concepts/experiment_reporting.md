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

Enable Trackio in your train config:

```yaml
trackio: true
trackio_project: eole
trackio_run_name: my-run-name  # optional
trackio_space_id: username/space_id  # optional
trackio_bucket_id: my-bucket-id  # optional
```

Notes:

- `trackio_run_name`: optional run name (auto-generated if omitted)
- `trackio_space_id`: optional Hugging Face Trackio Space target
- `trackio_bucket_id`: optional Hugging Face bucket for persistence

If available, EOLE also logs model and training configuration to Trackio's run config.

At runtime, the following failure behavior happens:
- If Trackio is not installed, EOLE logs a warning and continues training.
- If Trackio initialization fails, EOLE logs a warning and continues training.
- If Trackio raises during runtime logging, EOLE logs a warning once and disables Trackio for the rest of the run.

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