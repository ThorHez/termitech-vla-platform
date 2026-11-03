import copy
import dataclasses
from contextlib import nullcontext
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters import build_launch, get_templates
from vla_adapters import norm_entry
from vla_adapters.openpi_entry import training_module
from vla_protocol import canonical_digest
from test_composed_processing_contract import profile as composed_profile


def statistics():
    return {"norm_stats": {key: {"mean": [1., 2.], "std": [1., 1.], "q01": [0., 1.], "q99": [2., 3.]}
                           for key in ("state", "actions")}}


def test_norm_template_uses_registered_openpi_python_without_gpu_model_or_old_norm(tmp_path):
    template = next(row for row in get_templates() if row["id"] == "openpi-norm")
    assert template["status"] == "AVAILABLE" and template["min_gpus"] == 0
    assert template["required_inputs"] == ["dataset"]
    profile = {"adapter": "openpi-sft", "code_dir": str(tmp_path), "python": sys.executable}
    inputs = {"dataset": {"id": "dataset", "kind": "DATASET", "path": str(tmp_path / "dataset")}}
    launch = build_launch(profile, "openpi-norm", inputs, {}, str(tmp_path / "output"), [], recipe="my_recipe")
    assert launch["argv"][0] == sys.executable
    assert launch["argv"][2].endswith("norm_entry.py")
    spec = json.loads(launch["argv"][-1])
    assert spec["parameters"] == {"batch_size": 32, "num_workers": 0}
    assert launch["env"]["CUDA_VISIBLE_DEVICES"] == "" and launch["env"]["JAX_PLATFORMS"] == "cpu"
    with pytest.raises(ValueError, match="NORM_GPU_UNSUPPORTED"):
        build_launch(profile, "openpi-norm", inputs, {}, str(tmp_path), ["GPU-1"], recipe="my_recipe")
    with pytest.raises(ValueError, match="Unknown training parameters"):
        build_launch(profile, "openpi-norm", inputs, {"steps": 20}, str(tmp_path), [], recipe="my_recipe")


def test_norm_resolver_uses_selected_data_and_clears_old_norm_without_mutating_recipe(tmp_path, monkeypatch):
    @dataclasses.dataclass
    class Assets:
        assets_dir: str
        asset_id: str

    @dataclasses.dataclass
    class ComposableDataConfig:
        repo_id: str
        assets: Assets
        composition: dict = dataclasses.field(default_factory=lambda: composed_profile()["processing"])

    @dataclasses.dataclass
    class Config:
        data: ComposableDataConfig
        model: object = dataclasses.field(default_factory=lambda: SimpleNamespace(action_horizon=16))
        batch_size: int = 8
        num_workers: int = 8

    module = ModuleType("openpi.training.config")
    module.__file__ = str(tmp_path / "src/openpi/training/config.py")
    module.AssetsConfig = Assets
    base = Config(ComposableDataConfig("/old/dataset", Assets("/old/norm", "old")))
    original = copy.deepcopy(base)
    module.get_config = lambda _: base
    openpi = ModuleType("openpi")
    training = ModuleType("openpi.training")
    training.config = module
    openpi.training = training
    for name, value in [("openpi", openpi), ("openpi.training", training), ("openpi.training.config", module)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(norm_entry, "configure_cache", lambda _: None)
    monkeypatch.setattr(norm_entry, "install_registered_fps", lambda _: None)
    (tmp_path / "data/meta").mkdir(parents=True)
    (tmp_path / "data/meta/info.json").write_text("{}")
    spec = {"code_dir": str(tmp_path), "recipe": "recipe", "inputs": {"dataset": {"path": str(tmp_path / "data")}},
            "output_dir": str(tmp_path / "output"), "parameters": {"batch_size": 32, "num_workers": 0}}
    config = norm_entry.prepare_config(spec)
    assert config.data.repo_id == str(tmp_path / "data")
    assert config.data.assets.assets_dir == str(tmp_path / "output/.unbound")
    assert config.batch_size == 32 and config.num_workers == 0
    assert config.data.composition["cameras"] == []
    assert base == original


@pytest.fixture
def norm_job(tmp_path, monkeypatch):
    (tmp_path / "code/scripts").mkdir(parents=True)
    script = tmp_path / "code/scripts/compute_norm_stats.py"
    script.write_text("# registered statistics entrypoint\n")
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "data").mkdir(); (dataset / "meta").mkdir()
    (dataset / "data/frames.parquet").write_text("original")
    (dataset / "meta/info.json").write_text('{}')
    output = tmp_path / "output"
    output.mkdir()
    spec = {"template_id": "openpi-norm", "code_dir": str(tmp_path / "code"), "recipe": "recipe",
            "inputs": {"dataset": {"id": "dataset-id", "kind": "DATASET", "path": str(dataset)}},
            "output_dir": str(output), "parameters": {"batch_size": 32, "num_workers": 0}, "gpu_count": 0,
            "allowed_roots": [str(tmp_path)]}
    plan = {"generated_config": norm_entry.generated_config(spec), "inputs": {"dataset": str(dataset)},
            "datasets": [{"manifest_sha256": norm_entry.norm_manifest(dataset, spec)[2]}]}
    calls = []

    def main(config_file=None, output_dir=None, overwrite=False, numeric_only=False):
        assert Path(config_file).read_text() == plan["generated_config"]["source"]
        assert not overwrite
        assert numeric_only
        calls.append(output_dir)
        (Path(output_dir) / "norm_stats.json").write_text(json.dumps(statistics()))

    monkeypatch.setattr(norm_entry, "registered_fps_context", lambda _: nullcontext())
    monkeypatch.setattr(norm_entry, "prepare_config", lambda _: object())
    def statistics_module(*args, module_name):
        assert module_name == "compute_norm_stats"
        return nullcontext(SimpleNamespace(main=main))

    monkeypatch.setattr(norm_entry, "training_module", statistics_module)
    monkeypatch.setattr(norm_entry, "inspect_config", lambda *args: plan)
    return spec, plan, calls


def test_norm_preflight_does_not_scan_all_frames_or_publish_and_execution_publishes_norm(norm_job):
    spec, plan, calls = norm_job
    result = norm_entry.run_spec(spec, validate_only=True)
    assert result["plan"] == plan and not calls
    assert list(Path(spec["output_dir"]).iterdir()) == []
    # Different run output directories retain the same configuration identity.
    assert norm_entry.generated_config({**spec, "output_dir": "/other/output"}) == plan["generated_config"]
    norm_entry.run_spec({**spec, "expected_plan_digest": canonical_digest(plan)})
    manifest = json.loads((Path(spec["output_dir"]) / "artifacts.json").read_text())
    assert calls == [spec["output_dir"]]
    assert manifest["outputs"][0]["kind"] == "NORM"
    assert manifest["outputs"][0]["metadata"]["source_dataset_asset_id"] == "dataset-id"
    assert manifest["outputs"][0]["name"] == "data norm_stats"
    assert manifest["outputs"][0]["metadata"]["source_dataset_name"] == "data"
    assert manifest["outputs"][0]["metadata"]["coverage"] == "full"


@pytest.mark.parametrize("name,expected", [("  collect_1030  ", "collect_1030"),
    ("中文采集😀", "中文采集😀"), ("\n\x00\u202e\ud800", "data"), ("", "data"), (None, "data")])
def test_statistics_artifact_uses_safe_dataset_label_instead_of_recipe(norm_job, name, expected):
    spec, plan, _ = norm_job
    spec["inputs"]["dataset"]["name"] = name
    plan["generated_config"] = norm_entry.generated_config(spec)
    norm_entry.run_spec({**spec, "expected_plan_digest": canonical_digest(plan)})
    artifact = json.loads((Path(spec["output_dir"]) / "artifacts.json").read_text())["outputs"][0]
    assert artifact["name"] == expected + " norm_stats"
    assert artifact["metadata"]["source_dataset_name"] == expected
    assert artifact["metadata"]["source_dataset_asset_id"] == "dataset-id"
    artifact["name"].encode("utf-8", errors="strict")


def test_norm_names_cap_unicode_characters_and_legacy_paths_ignore_trailing_slashes():
    name = norm_entry.norm_artifact_name({"name": "数据😀" * 100, "path": "/data/fallback"})
    assert len(name) == 200 and name.endswith(" norm_stats")
    name.encode("utf-8", errors="strict")
    assert norm_entry.norm_artifact_name({"path": "/data/collect_1030///"}) == "collect_1030 norm_stats"
    assert norm_entry.norm_artifact_name({"path": "/", "id": "dataset-id"}) == "dataset-id norm_stats"


def test_changed_plan_cannot_start_statistics_or_create_outputs(norm_job):
    spec, plan, calls = norm_job
    with pytest.raises(ValueError, match="ASSETS_CHANGED_SINCE_PREFLIGHT"):
        norm_entry.run_spec({**spec, "expected_plan_digest": "f" * 64})
    assert not calls and not list(Path(spec["output_dir"]).iterdir())


def test_statistics_manifest_never_walks_or_tracks_unrelated_visual_files(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "data").mkdir(parents=True); (dataset / "meta").mkdir(); (dataset / "videos").mkdir()
    (dataset / "data/frames.parquet").write_bytes(b"numeric")
    (dataset / "meta/info.json").write_text('{}')
    video = dataset / "videos/corrupt.mp4"; video.write_bytes(b"invalid-video")
    manifest = norm_entry.file_manifest

    def only_numeric(path, spec):
        assert path == dataset / "data"
        return manifest(path, spec)

    monkeypatch.setattr(norm_entry, "file_manifest", only_numeric)
    first = norm_entry.norm_manifest(dataset, {"allowed_roots": [str(tmp_path)]})
    video.write_bytes(b"changed irrelevant visual data")
    assert norm_entry.norm_manifest(dataset, {"allowed_roots": [str(tmp_path)]}) == first
    (dataset / "meta/info.json").write_text('{"fps":30}')
    assert norm_entry.norm_manifest(dataset, {"allowed_roots": [str(tmp_path)]})[2] != first[2]


def test_unknown_legacy_processor_is_rejected_without_invoking_visual_create(tmp_path, monkeypatch):
    @dataclasses.dataclass
    class Assets:
        assets_dir: str
        asset_id: str

    @dataclasses.dataclass
    class LegacyFactory:
        repo_id: str
        assets: Assets

        def create(self, *args):
            pytest.fail("Unsupported legacy factories must never create a visual pipeline")

    module = ModuleType("openpi.training.config")
    module.__file__ = str(tmp_path / "config.py")
    module.AssetsConfig = Assets
    module.get_config = lambda _: SimpleNamespace(data=LegacyFactory("old", Assets("old", ".")))
    training = ModuleType("openpi.training"); training.config = module
    openpi = ModuleType("openpi"); openpi.training = training
    for name, value in [("openpi", openpi), ("openpi.training", training), ("openpi.training.config", module)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(norm_entry, "configure_cache", lambda _: None)
    dataset = tmp_path / "dataset"; (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{}')
    spec = {"code_dir": str(tmp_path), "recipe": "legacy", "inputs": {"dataset": {"path": str(dataset)}},
            "output_dir": str(tmp_path / "output"), "parameters": {"batch_size": 32, "num_workers": 0}}
    with pytest.raises(ValueError, match="NORM_NUMERIC_ONLY_UNSUPPORTED"):
        norm_entry.prepare_config(spec)


def test_statistics_dataset_and_collator_are_importable_in_spawn_workers(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "compute_norm_stats.py"
    contents = (
        "class _NumericDataset:\n"
        "    def __init__(self, values):\n"
        "        self.values = values\n"
        "    def __getitem__(self, index):\n"
        "        return self.values[index]\n"
        "def _numeric_collate(sample):\n"
        "    return sum(sample)\n"
        "def measure(dataset):\n"
        "    return _numeric_collate([dataset[0], dataset[1]])\n"
    ).encode()
    script.write_bytes(contents)
    previous_module = sys.modules.get("compute_norm_stats")
    previous_path = list(sys.path)
    with training_module(script, tmp_path, hashlib.sha256(contents).hexdigest(),
                         module_name="compute_norm_stats") as statistics_module:
        dataset = statistics_module._NumericDataset([2, 3])
        with multiprocessing.get_context("spawn").Pool(1) as workers:
            result = workers.apply_async(statistics_module.measure, (dataset,))
            assert result.get(timeout=20) == 5
    assert sys.path == previous_path
    assert sys.modules.get("compute_norm_stats") is previous_module


@pytest.mark.parametrize("change", ["missing_actions", "nan", "negative_std", "different_width", "reversed_quantiles"])
def test_invalid_statistics_cannot_be_registered(tmp_path, change):
    value = statistics()
    if change == "missing_actions":
        del value["norm_stats"]["actions"]
    elif change == "nan":
        value["norm_stats"]["state"]["mean"][0] = float("nan")
    elif change == "negative_std":
        value["norm_stats"]["state"]["std"][0] = -1
    elif change == "different_width":
        value["norm_stats"]["state"]["q99"] = [1]
    else:
        value["norm_stats"]["state"]["q01"][0] = 10
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, KeyError)):
        norm_entry.verify_statistics(path)
