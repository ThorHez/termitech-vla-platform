import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_adapters import build_launch, validate_parameters
from vla_adapters.openpi_entry import resolve_config


def test_no_shell_or_unknown_parameters(tmp_path):
    (tmp_path / "smoke_training.py").write_text("pass")
    profile = {"adapter": "smoke", "code_dir": str(tmp_path), "python": "/usr/bin/python3"}
    launch = build_launch(profile, "smoke", {}, {"steps": 3}, str(tmp_path / "with spaces"), [])
    assert launch["argv"][0] == "/usr/bin/python3"
    assert str(tmp_path / "with spaces") in launch["argv"]
    with pytest.raises(ValueError):
        build_launch(profile, "smoke", {}, {"command": "rm -rf /"}, str(tmp_path), [])


@pytest.mark.parametrize("value", [True, 1.5, 0, -1, float("nan"), float("inf")])
def test_reject_invalid_steps(value):
    with pytest.raises(ValueError):
        validate_parameters("smoke", {"steps": value})


def test_unvalidated_recipe_cannot_launch(tmp_path):
    profile = {"adapter": "openpi-sft", "code_dir": str(tmp_path), "python": "/usr/bin/python3", "recipe": "valid_recipe"}
    with pytest.raises(ValueError, match="PROFILE_NOT_VALIDATED"):
        build_launch(profile, "openpi-sft", {}, {}, str(tmp_path), ["GPU-a"])


def test_norm_file_must_be_the_actual_statistics_file(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    unrelated = tmp_path / "other.json"
    unrelated.write_text("{}")
    (tmp_path / "norm_stats.json").write_text("{}")
    spec = {"inputs": {"dataset": {"path": str(dataset)}, "norm": {"path": str(unrelated)},
                       "model": {"path": str(tmp_path)}}}
    with pytest.raises(ValueError, match="NORM_FORMAT"):
        resolve_config(None, spec, None, None)


def test_resolve_explicit_assets_never_reuses_recipe_paths(tmp_path):
    @dataclasses.dataclass
    class Data:
        repo_id: str = "/old/dataset"
        assets: object = None

    @dataclasses.dataclass
    class Schedule:
        peak_lr: float = 0.001

    @dataclasses.dataclass
    class Config:
        data: object
        lr_schedule: object
        weight_loader: object = None
        num_train_steps: int = 10
        batch_size: int = 8
        fsdp_devices: int = 8
        checkpoint_base_dir: str = "/old/checkpoints"
        exp_name: str = "old"
        resume: bool = True
        overwrite: bool = True
        wandb_enabled: bool = True

    dataset, norm, model = [tmp_path / key for key in ("dataset", "norm", "model")]
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}")
    norm.mkdir()
    (norm / "norm_stats.json").write_text("{}")
    (model / "params").mkdir(parents=True)
    base = Config(Data(), Schedule())
    spec = {"inputs": {key: {"path": str(value)} for key, value in
            {"dataset": dataset, "norm": norm, "model": model}.items()},
            "parameters": {"steps": 50, "batch_size": 4, "learning_rate": 0.0001},
            "gpu_count": 2, "output_dir": str(tmp_path / "output")}
    cfg = resolve_config(base, spec, SimpleNamespace(AssetsConfig=lambda **kw: kw),
                         SimpleNamespace(CheckpointWeightLoader=lambda path: path))
    assert cfg.data.repo_id == str(dataset)
    assert cfg.weight_loader == str(model / "params")
    assert cfg.data.assets == {"assets_dir": str(norm), "asset_id": "."}
    assert cfg.fsdp_devices == 2 and cfg.batch_size == 4
    assert not cfg.resume and not cfg.overwrite and not cfg.wandb_enabled
    assert base.data.repo_id == "/old/dataset"
