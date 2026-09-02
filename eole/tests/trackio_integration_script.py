"""Subprocess payload for eole/tests/test_trackio_integration.py.

Runs trackio in a fresh process so TRACKIO_DIR is set before trackio is
imported. Exits with code 77 (skip) when optional eole[trackio] dependencies
are missing, 0 on success.
"""

import os
import pathlib
import sys
import uuid

try:
    import psutil  # noqa: F401
    import trackio
except ImportError as exc:
    print(f"SKIP: optional trackio dependency missing: {exc}")
    sys.exit(77)

import yaml

from trackio import Api
from trackio.sqlite_storage import SQLiteStorage

from eole.config.run import TrainConfig
from eole.utils.report_manager import build_report_manager
from eole.utils.statistics import Statistics


project = "eole-trackio-it-" + uuid.uuid4().hex[:8]
run_name = "run"
workdir = pathlib.Path(os.environ["TRACKIO_IT_WORKDIR"])
# A realistic, hyphenated non-default config filename so the real Trackio backend
# exercises EOLE's derived artifact naming (hyphens + .yml suffix), not just "config.yaml".
original_config = workdir / "eole-integration-test.yml"
original_text = "trackio: true\ntrackio_project: placeholder\n"
original_config.write_text(original_text, encoding="utf-8")

config = TrainConfig(
    src_vocab="src.vocab",
    tgt_vocab="tgt.vocab",
    data={},
    model={"architecture": "rnn"},
    training={"batch_size": 2, "train_steps": 1},
    trackio=True,
    trackio_project=project,
    trackio_run_name=run_name,
    trackio_group="integration",
    trackio_auto_log_cpu=True,
    trackio_auto_log_gpu=False,
    trackio_system_log_interval=1.0,
    report_every=1,
)
config._config_file = str(original_config)

manager = None
try:
    manager = build_report_manager(config, gpu_rank=0)

    stats = Statistics(loss=2, n_batchs=1, n_sents=1, n_tokens=4, n_correct=3)
    manager.start()
    manager.report_training(1, 1, 0.1, None, stats)  # -> progress/* metrics
    manager.report_step(0.1, None, 1, train_stats=stats)  # -> train/* metrics
    trackio.log_cpu()

    download_root = workdir / "downloads"
    original_artifact = pathlib.Path(
        trackio.use_artifact("eole-integration-test.yml").download(download_root / "original")
    )
    effective_artifact = pathlib.Path(
        trackio.use_artifact("eole-integration-test-effective.yml").download(download_root / "effective")
    )
finally:
    if manager is not None:
        manager.close()

runs = Api().runs(project)
assert run_name in [r.name for r in runs], [r.name for r in runs]

run_config = next(r for r in runs if r.name == run_name).config
assert run_config["_Group"] == "integration", run_config
assert run_config["eole_version"]
assert "model" in run_config
assert "training" in run_config

metric_names = set(SQLiteStorage.get_all_metrics_for_run(project, run_name))
assert {"train/xent", "train/ppl", "train/lr"} <= metric_names, metric_names
assert {"progress/ppl", "progress/lr"} <= metric_names, metric_names

assert SQLiteStorage.has_system_metrics(project)
assert SQLiteStorage.get_all_system_metrics_for_run(project, run_name)

downloaded_original = original_artifact / "eole-integration-test.yml"
assert downloaded_original.read_text(encoding="utf-8") == original_text

effective_files = list(effective_artifact.iterdir())
assert len(effective_files) == 1, effective_files
assert effective_files[0].name == "eole-integration-test-effective.yml", effective_files
effective_config = yaml.safe_load(effective_files[0].read_text(encoding="utf-8"))
assert effective_config["trackio"] is True
assert effective_config["trackio_project"] == project
assert effective_config["training"]["batch_size"] == 2
# full runtime config: defaults included, non-JSON values stringified, private attrs excluded
assert effective_config["tensorboard"] is False
assert effective_config["training"]["save_format"] == "pytorch"
assert effective_config["training"]["compute_dtype"] == "torch.float32"
assert "_config_file" not in effective_config

print(f"OK (trackio {trackio.__version__})")
