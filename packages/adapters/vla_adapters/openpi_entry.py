"""Executed by the registered OpenPI Python, never imported by the control plane.

Supports the dataclass-based scripts/train.py interface. The administrator must
validate each recipe against its real assets before marking a profile validated.
"""

import argparse
import dataclasses
import importlib.util
import json
import logging
import math
from pathlib import Path


def resolve_config(base, spec, config_module, weight_loaders):
    """Explicit asset bindings override recipe defaults; never modify the recipe."""
    inputs = spec["inputs"]
    dataset = Path(inputs["dataset"]["path"]).resolve(strict=True)
    norm = Path(inputs["norm"]["path"]).resolve(strict=True)
    model = Path(inputs["model"]["path"]).resolve(strict=True)
    if not dataset.is_dir() or not (dataset / "meta" / "info.json").is_file():
        raise ValueError("DATASET_FORMAT: expected a local LeRobot dataset with meta/info.json")
    if norm.is_file() and norm.name != "norm_stats.json":
        raise ValueError("NORM_FORMAT: a file asset must point to norm_stats.json explicitly")
    norm_dir = norm.parent if norm.is_file() else norm
    if not (norm_dir / "norm_stats.json").is_file():
        raise ValueError("NORM_MISSING: norm asset must contain norm_stats.json")
    params_dir = model / "params" if (model / "params").is_dir() else model
    if not params_dir.is_dir():
        raise ValueError("MODEL_FORMAT: expected a local Orbax params directory")
    if not dataclasses.is_dataclass(base.data) or not hasattr(base.data, "repo_id"):
        raise ValueError("RECIPE_UNSUPPORTED: data factory must support a local repo_id")
    if not hasattr(base.lr_schedule, "peak_lr"):
        raise ValueError("RECIPE_UNSUPPORTED: learning rate schedule has no peak_lr")
    data = dataclasses.replace(base.data, repo_id=str(dataset), assets=config_module.AssetsConfig(
        assets_dir=str(norm_dir), asset_id="."))
    schedule = dataclasses.replace(base.lr_schedule, peak_lr=spec["parameters"]["learning_rate"])
    return dataclasses.replace(base, data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(params_dir)),
        lr_schedule=schedule, num_train_steps=spec["parameters"]["steps"],
        batch_size=spec["parameters"]["batch_size"], fsdp_devices=spec["gpu_count"],
        checkpoint_base_dir=str(Path(spec["output_dir"]) / "checkpoints"),
        exp_name="platform", resume=False, overwrite=False, wandb_enabled=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec_json)
    # Import through the existing environment. Agent separately verifies this origin.
    import openpi.training.config as config_module
    import openpi.training.weight_loaders as weight_loaders
    import wandb

    code = Path(spec["code_dir"]).resolve(strict=True)
    entry = (code / "scripts" / "train.py").resolve(strict=True)
    if not entry.is_relative_to(code):
        raise ValueError("ENTRYPOINT_OUTSIDE_CODE_DIR")
    config = resolve_config(config_module.get_config(spec["recipe"]), spec, config_module, weight_loaders)
    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_training.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))

    def log_metrics(values, step=None, **kwargs):
        metrics = {key: float(value) for key, value in values.items() if hasattr(value, "__float__")}
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("NON_FINITE_METRIC: training reported NaN or infinity")
        print("VLA_EVENT:" + json.dumps({"type": "METRIC", "data": {
            **metrics, "step": step, "total_steps": config.num_train_steps}}), flush=True)

    logging.basicConfig(level=logging.INFO)
    module_spec = importlib.util.spec_from_file_location("vla_registered_openpi_train", entry)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # This script calls wandb.log; retain local metrics while disabling remote tracking.
    # init() may replace wandb.log, so patch immediately after initialization too.
    original_init = module.init_wandb

    def local_init(*args, **kwargs):
        result = original_init(*args, **kwargs)
        wandb.log = log_metrics
        return result

    module.init_wandb = local_init
    module.main(config)
    checkpoints = sorted((p for p in Path(config.checkpoint_dir).iterdir()
                          if p.is_dir() and p.name.isdigit() and (p / "params").is_dir()),
                         key=lambda p: int(p.name))
    if not checkpoints:
        raise RuntimeError("OUTPUT_MISSING: training exited without a completed params checkpoint")
    last = checkpoints[-1]
    artifact = {"kind": "MODEL", "name": spec["recipe"] + " / " + last.name,
                "path": str(last), "version": last.name,
                "metadata": {"format": "openpi-orbax", "recipe": spec["recipe"],
                             "step": int(last.name), "evaluated": False}}
    (output / "artifacts.json").write_text(json.dumps([artifact], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
