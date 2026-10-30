import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vla_adapters import openpi_entry, training_plan
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.training_plan import fingerprint, inspect_training_config, training_entrypoint


@dataclasses.dataclass
class Assets:
    assets_dir: str
    asset_id: str = "."


@dataclasses.dataclass
class Data:
    repo_id: str
    assets: object = None


@dataclasses.dataclass
class Multi(Data):
    datasets: list = dataclasses.field(default_factory=list)
    weights: list = dataclasses.field(default_factory=lambda: [1.0, 20.0, 50.0])
    state_pad_dim: int = 32
    use_merged_norm_stats: bool = True


@dataclasses.dataclass
class Schedule:
    peak_lr: float = 3e-5


@dataclasses.dataclass
class Config:
    data: object
    lr_schedule: object = dataclasses.field(default_factory=Schedule)
    weight_loader: object = None
    num_train_steps: int = 60000
    batch_size: int = 72
    fsdp_devices: int = 8
    checkpoint_base_dir: str = "/old/checkpoints"
    exp_name: str = "old"
    resume: bool = True
    overwrite: bool = True
    wandb_enabled: bool = True
    save_interval: int = 1000
    keep_period: int = 30000


@pytest.fixture
def training(tmp_path, monkeypatch):
    # These fixture files exercise plan assembly, not Parquet decoding. Actual
    # timestamp reads and failure cases have real-file tests in their own suite.
    monkeypatch.setattr(training_plan, "inspect_dataset_timestamps", lambda data, spec: {
        "schema_version": 1, "status": "verified", "checked_rows": data["frames"],
        "effective_fps": data.get("effective_fps"), "data_manifest_digest": data["data_manifest_digest"]})
    monkeypatch.setattr(training_plan, "inspect_composed_samples_isolated", lambda factory, config, spec, data: {
        "schema_version": 1, "status": "verified" if "composition" in data else "not_applicable"})
    cache = str(tmp_path / "training-cache")
    monkeypatch.setattr(training_plan, "inspect_training_cache", lambda datasets, spec: {
        "datasets_cache": cache, "paths": {"HF_DATASETS_CACHE": cache},
        "builders": [{"path": row["path"], "cache_dir": cache + "/parquet/" + str(i)}
                     for i, row in enumerate(datasets)]})
    factories = []
    for name in ("old", "new1", "new2"):
        path = tmp_path / name
        (path / "meta").mkdir(parents=True)
        (path / "meta/info.json").write_text(json.dumps({"total_frames": 20, "total_episodes": 2,
            "features": {"actions": {"shape": [16, 10]}}}))
        (path / "data").mkdir()
        (path / "data/episode_000000.parquet").write_bytes(b"PAR1-dataset-fixture-PAR1")
        (path / "norm_stats.json").write_text(json.dumps({"norm_stats": {
            role: {"mean": [0., 0.], "std": [1., 1.], "q01": [-1., -1.], "q99": [1., 1.]}
            for role in ("state", "actions")}}))
        factories.append(Data(str(path), Assets(str(path))))
    for entry in ("train.py", "train_multi_dataset.py"):
        path = tmp_path / "code/scripts" / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('preflight must not execute training entry')\n"
                        "def main(config): pass\ndef init_wandb(*a, **kw): pass\n")
    params = tmp_path / "model/params"
    params.mkdir(parents=True)
    (params / "_METADATA").write_text('{"tree_metadata": {}}')
    (params / "manifest.ocdbt").write_bytes(b"model-parameter-manifest")
    (params / "d").mkdir()
    (params / "d/shard").write_bytes(b"weights")
    (tmp_path / "model/train_state").mkdir()
    (tmp_path / "model/train_state/optimizer").write_bytes(b"state" * 30)
    base = Config(Multi(repo_id="multi", datasets=factories))
    spec = {"recipe": "umi_three_datasets", "code_dir": str(tmp_path / "code"),
            "inputs": {"model": {"id": "selected-model", "path": str(tmp_path / "model")},
                       **{f"dataset_{i}": {"id": f"dataset-{i}", "path": factory.repo_id} for i, factory in enumerate(factories)},
                       **{f"norm_{i}": {"id": f"norm-{i}", "path": str(Path(factory.repo_id) / "norm_stats.json")} for i, factory in enumerate(factories)}},
            "dataset_bindings": [{"dataset_asset_id": f"dataset-{i}", "norm_asset_id": f"norm-{i}",
                                  "recipe_index": i, "weight": weight} for i, weight in enumerate([1., 20., 50.])],
            "parameters": {"steps": 20, "batch_size": 72, "learning_rate": 3e-5},
            "gpu_count": 8, "allowed_roots": [str(tmp_path)], "output_dir": str(tmp_path / "outputs")}
    module = SimpleNamespace(AssetsConfig=Assets, MultiDataConfigFactory=Multi)
    loaders = SimpleNamespace(CheckpointWeightLoader=lambda p: p)
    return base, spec, module, loaders


def resolve(training):
    return resolve_config(*training)


def test_multi_binds_all_selected_assets_and_routes_actual_entry(training):
    base, spec, module, loaders = training
    original = copy.deepcopy(base)
    config = resolve_config(base, spec, module, loaders)
    assert config.data.datasets[:2] == base.data.datasets[:2]
    assert config.data.weights == [1., 20., 50.]
    assert config.data.state_pad_dim == 32 and config.data.use_merged_norm_stats
    assert config.data.datasets[2].repo_id == spec["inputs"]["dataset_2"]["path"]
    assert base == original
    checks = []
    plan = inspect_training_config(config, spec, checks=checks)
    assert plan["entrypoint"].endswith("train_multi_dataset.py")
    assert [d["path"] for d in plan["datasets"]] == [d.repo_id for d in original.data.datasets]
    assert len(checks) == 8 and all(c["ok"] for c in checks)
    assert any(c["name"] == "dataset_timestamps" for c in checks)
    assert all(data["timestamp_contract"]["status"] == "verified" for data in plan["datasets"])


def test_real_timestamp_failure_is_reported_before_model_or_cache_validation(training, monkeypatch):
    config = resolve(training)
    checks = []

    def mismatch(data, spec):
        raise ValueError("DATASET_TIMESTAMPS_INVALID: 生效 20 Hz，实测帧间隔 0.016667 秒")

    monkeypatch.setattr(training_plan, "inspect_dataset_timestamps", mismatch)
    monkeypatch.setattr(training_plan, "model_contract", lambda *_: pytest.fail("Do not continue after invalid timestamps"))
    with pytest.raises(ValueError, match="DATASET_TIMESTAMPS_INVALID"):
        inspect_training_config(config, training[1], checks=checks)
    assert checks[-1]["name"] == "dataset_timestamps" and checks[-1]["ok"] is False


def test_decoded_sample_failure_blocks_before_model_or_cache_validation(training, monkeypatch):
    config = resolve(training)
    checks = []
    def mismatch(*args):
        raise ValueError("DATASET_SAMPLE_INVALID: sample=0 camera=front actual=[3,224,224] declared=hwc")
    monkeypatch.setattr(training_plan, "inspect_composed_samples_isolated", mismatch)
    monkeypatch.setattr(training_plan, "model_contract", lambda *_: pytest.fail("Do not continue after a sample failure"))
    with pytest.raises(ValueError, match="DATASET_SAMPLE_INVALID"):
        inspect_training_config(config, training[1], checks=checks)
    assert checks[-1]["name"] == "dataset_samples" and checks[-1]["ok"] is False


def test_single_dataset_binding_uses_selected_dataset_and_single_entry(training):
    base, spec, module, loaders = training
    base.data = base.data.datasets[0]
    spec["inputs"] = {"dataset": spec["inputs"]["dataset_2"], "norm": spec["inputs"]["norm_2"], "model": spec["inputs"]["model"]}
    spec.pop("dataset_bindings")
    config = resolve_config(base, spec, module, loaders)
    assert config.data.repo_id == spec["inputs"]["dataset"]["path"]
    assert training_entrypoint(config, spec).name == "train.py"


@pytest.mark.parametrize("count", [1, 2])
def test_rebound_multi_recipe_keeps_compatible_entry_for_actual_selected_count(training, count):
    base, spec, module, loaders = training
    original = copy.deepcopy(base)
    spec["dataset_bindings"] = spec["dataset_bindings"][:count]
    roles = {"model", *(f"dataset_{i}" for i in range(count)), *(f"norm_{i}" for i in range(count))}
    spec["inputs"] = {key: value for key, value in spec["inputs"].items() if key in roles}
    config = resolve_config(base, spec, module, loaders)
    plan = inspect_training_config(config, spec, checks=[])
    assert len(config.data.datasets) == len(plan["datasets"]) == count
    assert config.data.weights == [row["weight"] for row in spec["dataset_bindings"]]
    # A one-child MultiDataConfigFactory still needs the multi loader; the
    # effective container cannot be consumed by the built-in single loader.
    assert plan["entrypoint"] == str(Path(spec["code_dir"]) / "scripts/train_multi_dataset.py")
    assert base == original


@pytest.mark.parametrize("case", ["one_match", "no_match", "multiple_matches"])
def test_multi_requires_explicit_bindings_even_if_one_default_path_matches(training, case):
    base, spec, module, loaders = training
    spec["inputs"] = {"dataset": spec["inputs"]["dataset_2"], "norm": spec["inputs"]["norm_2"], "model": spec["inputs"]["model"]}
    spec.pop("dataset_bindings")
    if case == "no_match":
        base.data.datasets = base.data.datasets[:2]
    elif case == "multiple_matches":
        base.data.datasets[0] = base.data.datasets[2]
    with pytest.raises(ValueError, match="DATASET_BINDINGS_REQUIRED"):
        resolve_config(base, spec, module, loaders)
    with pytest.raises(ValueError, match="DATASET_BINDINGS_REQUIRED"):
        inspect_training_config(base, spec, checks=[])


def test_every_child_must_be_under_agent_roots(training):
    base, spec, module, loaders = training
    spec["allowed_roots"] = [spec["inputs"]["dataset_2"]["path"], spec["inputs"]["model"]["path"]]
    with pytest.raises(ValueError, match="PATH_OUTSIDE_ROOTS"):
        resolve_config(base, spec, module, loaders)


@pytest.mark.parametrize("case,error", [("empty_data", "DATASET_FORMAT"), ("invalid_metadata", "DATASET_FORMAT"),
    ("nonfinite_norm", "NORM_FORMAT"), ("missing_norm", "No such file"), ("empty_model", "MODEL_FORMAT"),
    ("weights", "RECIPE_UNSUPPORTED"), ("bad_entry", "ENTRYPOINT_UNSUPPORTED")])
def test_contract_rejects_invalid_secondary_assets_and_entry(training, case, error):
    base, spec, module, loaders = training
    old = Path(base.data.datasets[0].repo_id)
    if case == "empty_data":
        (old / "data/episode_000000.parquet").unlink()
    elif case == "invalid_metadata":
        (old / "meta/info.json").write_text('{}')
    elif case == "nonfinite_norm":
        (old / "norm_stats.json").write_text('{"norm_stats":{"state":{"mean":["bad"],"std":[1]},"actions":{}}}')
    elif case == "missing_norm":
        (old / "norm_stats.json").unlink()
    elif case == "empty_model":
        for file in (Path(spec["inputs"]["model"]["path"]) / "params").rglob('*'):
            if file.is_file(): file.unlink()
    elif case == "bad_entry":
        (Path(spec["code_dir"]) / "scripts/train_multi_dataset.py").write_text('pass')
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        config = resolve_config(base, spec, module, loaders)
        if case == "weights":
            config.data.weights = [1., 2.]
        inspect_training_config(config, spec, checks=[])


def test_file_drift_changes_plan_but_output_directory_does_not(training):
    base, spec, module, loaders = training
    config = resolve_config(base, spec, module, loaders)
    first = inspect_training_config(config, spec, checks=[])
    spec["output_dir"] += "-different-attempt"
    second_config = resolve_config(base, spec, module, loaders)
    assert fingerprint(first) == fingerprint(inspect_training_config(second_config, spec, checks=[]))
    (Path(base.data.datasets[1].repo_id) / "data/episode_000000.parquet").write_bytes(b"different dataset")
    assert fingerprint(first) != fingerprint(inspect_training_config(config, spec, checks=[]))


def test_preflight_cli_does_not_execute_training_or_create_output(training, monkeypatch, capsys):
    base, spec, module, loaders = training
    config = resolve_config(base, spec, module, loaders)
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda s: (config, None))
    monkeypatch.setattr(sys, "argv", ["openpi_entry", "--spec-json", json.dumps(spec), "--validate-only"])
    openpi_entry.main()
    result = json.loads(capsys.readouterr().out.split("VLA_PREFLIGHT:")[-1])
    assert result["scope"] == "configuration_and_local_assets" and "error" not in result
    assert not Path(spec["output_dir"]).exists()


def test_cache_permission_failure_blocks_preflight_before_training(training, monkeypatch, capsys):
    base, spec, module, loaders = training
    config = resolve_config(base, spec, module, loaders)

    def denied(*args):
        raise ValueError("TRAINING_CACHE_UNWRITABLE: root-owned dataset-cache.lock")

    monkeypatch.setattr(training_plan, "inspect_training_cache", denied)
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda s: (config, None))
    monkeypatch.setattr(sys, "argv", ["openpi_entry", "--spec-json", json.dumps(spec), "--validate-only"])
    with pytest.raises(SystemExit) as failure:
        openpi_entry.main()
    assert failure.value.code == 2
    result = json.loads(capsys.readouterr().out.split("VLA_PREFLIGHT:")[-1])
    assert result["error"]["code"] == "TRAINING_CACHE_UNWRITABLE"
    assert "dataset-cache.lock" in result["error"]["message"]
    assert result["checks"][-1]["name"] == "training_cache" and result["checks"][-1]["ok"] is False
    assert not Path(spec["output_dir"]).exists()


def test_real_wrapper_refuses_unfrozen_or_changed_plan_before_training(training, monkeypatch):
    base, spec, module, loaders = training
    config = resolve_config(base, spec, module, loaders)
    spec["expected_plan_digest"] = "0" * 64
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda s: (config, None))
    monkeypatch.setattr(sys, "argv", ["openpi_entry", "--spec-json", json.dumps(spec)])
    with pytest.raises(ValueError, match="ASSETS_CHANGED_SINCE_PREFLIGHT"):
        openpi_entry.main()
    assert not Path(spec["output_dir"]).exists()


def test_retained_checkpoints_increase_storage_budget(training):
    base, spec, module, loaders = training
    short = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    spec["parameters"]["steps"] = 60000
    full = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert short["storage"]["checkpoint_copies"] == 1
    assert full["storage"]["checkpoint_copies"] == 3
    assert full["storage"]["required_bytes"] > short["storage"]["required_bytes"]


def test_metadata_sidecars_are_not_model_weights(training):
    base, spec, module, loaders = training
    params = Path(spec["inputs"]["model"]["path"]) / "params"
    (params / "d/shard").unlink()
    (params / "_sharding").write_text('{"only":"metadata"}')
    with pytest.raises(ValueError, match="MODEL_FORMAT"):
        inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])


@pytest.mark.parametrize("field", ["mean", "std"])
def test_required_norm_arrays_cannot_be_null(training, field):
    base, spec, module, loaders = training
    norm = Path(base.data.datasets[0].repo_id) / "norm_stats.json"
    document = json.loads(norm.read_text())
    document["norm_stats"]["state"][field] = None
    norm.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="NORM_FORMAT"):
        inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])


def test_video_assets_must_exist_and_are_in_frozen_plan(training):
    base, spec, module, loaders = training
    dataset = Path(base.data.datasets[0].repo_id)
    info_path = dataset / "meta/info.json"
    document = json.loads(info_path.read_text())
    document.update(video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    document["features"]["camera"] = {"dtype": "video", "shape": [224, 224, 3]}
    info_path.write_text(json.dumps(document))
    videos = dataset / "videos/chunk-000/camera"
    videos.mkdir(parents=True)
    config = resolve_config(base, spec, module, loaders)
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_training_config(config, spec, checks=[])
    for i in range(2):
        (videos / f"episode_{i:06d}.mp4").write_bytes(b"video")
    first = inspect_training_config(config, spec, checks=[])
    (videos / "episode_000001.mp4").write_bytes(b"different video")
    assert fingerprint(first) != fingerprint(inspect_training_config(config, spec, checks=[]))


def test_small_reference_checkpoint_does_not_remove_optimizer_space_allowance(training):
    from vla_adapters.training_plan import storage_estimate
    base, *_ = training
    estimate = storage_estimate(base, {"parameter_bytes": 10 * 1024 ** 3}, 20 * 1024 ** 3)
    assert estimate["estimated_checkpoint_bytes"] >= 50 * 1024 ** 3


@pytest.mark.parametrize("value", ["/tmp/train.py", "../train.py", "scripts/../train.py", "./train.py", "scripts//train.py",
    "scripts\\train.py", "scripts/train.sh", "", "scripts/train.py\n", "scripts/\u200btrain.py", 7, "x" * 4097 + ".py"])
def test_entrypoint_requires_project_relative_python_file(value):
    from vla_adapters.entrypoints import validate_entrypoint
    with pytest.raises(ValueError, match="INVALID_ENTRYPOINT"):
        validate_entrypoint(value)


def test_explicit_custom_entrypoint_is_frozen_and_drift_changes_plan(training):
    base, spec, module, loaders = training
    selected = Path(spec["code_dir"]) / "scripts/custom_multi.py"
    selected.write_text("def main(config): pass\ndef init_wandb(*a, **k): pass\n")
    spec["entrypoint"] = "scripts/custom_multi.py"
    config = resolve_config(base, spec, module, loaders)
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["requested_entrypoint"] == "scripts/custom_multi.py"
    assert plan["entrypoint"] == str(selected)
    selected.write_text("def main(config): pass\ndef init_wandb(*a, **k): pass\n# changed\n")
    assert fingerprint(plan) != fingerprint(inspect_training_config(config, spec, checks=[]))


def test_multi_cannot_choose_known_single_dataset_entry(training):
    base, spec, module, loaders = training
    spec["entrypoint"] = "scripts/train.py"
    with pytest.raises(ValueError, match="ENTRYPOINT_RECIPE_MISMATCH"):
        training_entrypoint(resolve_config(base, spec, module, loaders), spec)


def test_internal_symlink_is_allowed_but_project_escape_is_rejected(training, tmp_path):
    base, spec, module, loaders = training
    code = Path(spec["code_dir"])
    spec["entrypoint"] = "scripts/alias.py"
    alias = code / spec["entrypoint"]
    alias.symlink_to(code / "scripts/train_multi_dataset.py")
    assert training_entrypoint(resolve_config(base, spec, module, loaders), spec).name == "train_multi_dataset.py"
    alias.unlink()
    outside = tmp_path / "outside.py"
    outside.write_text("def main(config): pass\ndef init_wandb(): pass\n")
    alias.symlink_to(outside)
    with pytest.raises(ValueError, match="ENTRYPOINT_OUTSIDE_CODE_DIR"):
        training_entrypoint(resolve_config(base, spec, module, loaders), spec)


def test_selected_file_must_exist_and_implement_supported_interface(training):
    base, spec, module, loaders = training
    spec["entrypoint"] = "scripts/missing.py"
    config = resolve_config(base, spec, module, loaders)
    with pytest.raises(ValueError, match="ENTRYPOINT_UNAVAILABLE"):
        training_entrypoint(config, spec)
    selected = Path(spec["code_dir"]) / spec["entrypoint"]
    selected.write_text("def main(config): pass\n")
    with pytest.raises(ValueError, match="ENTRYPOINT_UNSUPPORTED"):
        training_entrypoint(config, spec)


def test_real_wrapper_executes_selected_script_after_plan_check(training, monkeypatch):
    base, spec, module, loaders = training
    selected = Path(spec["code_dir"]) / "scripts/selected_multi.py"
    selected.write_text('''from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
@dataclass
class Marker:
    value: str
def init_wandb(*args, **kwargs): pass
def main(config):
    init_wandb(config)
    root = Path(config.checkpoint_base_dir).parent
    (root / "selected-script-ran").write_text(Marker("selected entry").value)
    (config.checkpoint_dir / "7" / "params").mkdir(parents=True)
''')
    spec["entrypoint"] = "scripts/selected_multi.py"
    config = resolve_config(base, spec, module, loaders)
    monkeypatch.setattr(Config, "checkpoint_dir", property(lambda c: Path(c.checkpoint_base_dir) / "platform"), raising=False)
    spec["expected_plan_digest"] = fingerprint(inspect_training_config(config, spec, checks=[]))
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda s: (config, None))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(log=None))
    monkeypatch.setattr(sys, "argv", ["openpi_entry", "--spec-json", json.dumps(spec)])
    openpi_entry.main()
    output = Path(spec["output_dir"])
    assert (output / "selected-script-ran").read_text() == "selected entry"
    effective = json.loads((output / "effective_training.json").read_text())
    assert effective["execution_plan"]["entrypoint"] == str(selected)
    assert effective["execution_plan"]["requested_entrypoint"] == spec["entrypoint"]
    assert json.loads((output / "artifacts.json").read_text())[0]["version"] == "7"


@pytest.mark.parametrize("signature", ["", "config, extra", "*, config", "config, *, required"])
def test_preflight_rejects_main_that_cannot_accept_one_config(training, signature):
    base, spec, module, loaders = training
    selected = Path(spec["code_dir"]) / "scripts/incompatible.py"
    selected.write_text(f"def main({signature}): pass\ndef init_wandb(): pass\n")
    spec["entrypoint"] = "scripts/incompatible.py"
    with pytest.raises(ValueError, match="ENTRYPOINT_UNSUPPORTED"):
        training_entrypoint(resolve_config(base, spec, module, loaders), spec)


def test_loading_uses_verified_source_even_with_matching_stale_pyc(tmp_path):
    import hashlib
    import py_compile
    script = tmp_path / "custom.py"
    old = b"def main(config): return 'old'\n"
    new = b"def main(config): return 'new'\n"
    script.write_bytes(old)
    timestamp = script.stat().st_mtime
    py_compile.compile(str(script), doraise=True)
    script.write_bytes(new)
    os.utime(script, (timestamp, timestamp))
    with openpi_entry.training_module(script, tmp_path, hashlib.sha256(new).hexdigest()) as module:
        assert module.main(None) == "new"
    with pytest.raises(ValueError, match="ASSETS_CHANGED_SINCE_PREFLIGHT"):
        with openpi_entry.training_module(script, tmp_path, hashlib.sha256(old).hexdigest()):
            raise AssertionError("changed source must not execute")
