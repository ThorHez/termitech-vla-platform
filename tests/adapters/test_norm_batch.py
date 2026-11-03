import ast
from contextlib import nullcontext
import copy
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from vla_adapters import build_launch, norm_entry
from vla_adapters.norm_datasets import normalize_norm_datasets, norm_inputs, validate_norm_assets
from vla_protocol import canonical_digest
from test_composed_processing_contract import profile as composed_profile


def statistics(index=0):
    return {"norm_stats": {key: {"mean": [float(index)], "std": [1.], "q01": [-1.], "q99": [3.]}
                           for key in ("state", "actions")}}


@pytest.mark.parametrize("invalid", [
    [], [{"dataset_asset_id": "a", "recipe_index": 0, "weight": 1}],
    [{"dataset_asset_id": "a", "recipe_index": 0, "norm_asset_id": "old"}],
    [{"dataset_asset_id": "a"}], [{"dataset_asset_id": "a", "recipe_index": True}],
    [{"dataset_asset_id": "a", "recipe_index": 0}, {"dataset_asset_id": "a", "recipe_index": 1}],
])
def test_independent_statistics_rows_reject_training_controls_and_duplicates(invalid):
    with pytest.raises(ValueError, match="NORM_DATASETS"):
        normalize_norm_datasets(invalid)


def test_batch_launch_contains_only_dataset_assets_and_freezes_processing(tmp_path):
    rows = [{"dataset_asset_id": "first", "recipe_index": 0}, {"dataset_asset_id": "second", "recipe_index": 1}]
    inputs = {role: {"id": asset, "kind": "DATASET", "path": str(tmp_path / asset)} for role, asset in norm_inputs(rows).items()}
    profile = {"adapter": "openpi-sft", "code_dir": str(tmp_path), "python": sys.executable}
    launch = build_launch(profile, "openpi-norm", inputs, {}, str(tmp_path / "out"), [], recipe="recipe", norm_datasets=rows)
    spec = json.loads(launch["argv"][-1])
    assert spec["norm_datasets"] == rows
    assert set(spec["inputs"]) == {"dataset_0", "dataset_1"}
    assert "dataset_bindings" not in spec and "model" not in spec["inputs"]
    with pytest.raises(ValueError, match="NORM_DATASETS"):
        validate_norm_assets(rows, {**inputs, "model": {"id": "model"}})
    changed = copy.deepcopy(inputs); changed["dataset_1"]["id"] = "other"
    with pytest.raises(ValueError, match="NORM_DATASETS"):
        validate_norm_assets(rows, changed)


@pytest.fixture
def batch_job(tmp_path, monkeypatch):
    code = tmp_path / "code"; (code / "scripts").mkdir(parents=True)
    (code / "scripts/compute_norm_stats.py").write_text("# registered entrypoint\n")
    output = tmp_path / "output"; output.mkdir()
    rows = [{"dataset_asset_id": "first", "recipe_index": 0}, {"dataset_asset_id": "second", "recipe_index": 0}]
    inputs = {}
    for index, row in enumerate(rows):
        dataset = tmp_path / f"data-{index}"; dataset.mkdir()
        (dataset / "data").mkdir(); (dataset / "meta").mkdir()
        (dataset / "data/frames.parquet").write_text("original")
        (dataset / "meta/info.json").write_text('{}')
        inputs[f"dataset_{index}"] = {"id": row["dataset_asset_id"], "kind": "DATASET", "path": str(dataset)}
        if index == 0:
            inputs[f"dataset_{index}"]["name"] = "collect_1030"
    spec = {"template_id": "openpi-norm", "code_dir": str(code), "recipe": "recipe", "inputs": inputs,
            "norm_datasets": rows, "output_dir": str(output), "parameters": {"batch_size": 32, "num_workers": 0},
            "gpu_count": 0, "allowed_roots": [str(tmp_path)]}
    datasets = [{"path": asset["path"], "manifest_sha256": norm_entry.norm_manifest(Path(asset["path"]), spec)[2]}
                for asset in inputs.values()]
    plan = {"generated_config": norm_entry.generated_config(spec), "inputs": {role: asset["path"] for role, asset in inputs.items()},
            "datasets": datasets}
    calls = []
    state = {"fail_second": False, "change_first": False}

    def prepare(selected):
        index = selected.get("norm_dataset_index", 0)
        return SimpleNamespace(index=index, dataset_id=selected["norm_datasets"][index]["dataset_asset_id"],
                               output_dir=selected["output_dir"])

    def main(config_file=None, output_dir=None, overwrite=False, numeric_only=False):
        assert numeric_only
        before = list(sys.path)
        try:
            module = {}; exec(compile(Path(config_file).read_text(), config_file, "exec"), module)
            selected = module["build_config"](str(Path(output_dir) / ".config-only-output"))
        finally:
            sys.path[:] = before
        calls.append(selected.dataset_id)
        if selected.index == 1 and state["fail_second"]:
            raise ValueError("second dataset failed")
        (Path(output_dir) / "norm_stats.json").write_text(json.dumps(statistics(selected.index)))
        if selected.index == 1 and state["change_first"]:
            (Path(inputs["dataset_0"]["path"]) / "data/frames.parquet").write_text("changed during second scan")

    monkeypatch.setattr(norm_entry, "registered_fps_context", lambda _: nullcontext())
    monkeypatch.setattr(norm_entry, "prepare_config", prepare)
    monkeypatch.setattr(norm_entry, "training_module", lambda *args, **kwargs: nullcontext(SimpleNamespace(main=main)))
    monkeypatch.setattr(norm_entry, "inspect_config", lambda *args: plan)
    return spec, plan, calls, state


def test_batch_statistics_compute_each_source_separately_and_register_exact_source_ids(batch_job):
    spec, plan, calls, _ = batch_job
    preview = norm_entry.run_spec(spec, validate_only=True)
    assert preview["plan"] == plan and calls == []
    norm_entry.run_spec({**spec, "expected_plan_digest": canonical_digest(plan)})
    output = Path(spec["output_dir"])
    outputs = json.loads((output / "artifacts.json").read_text())["outputs"]
    assert calls == ["first", "second"]
    assert [row["metadata"]["source_dataset_asset_id"] for row in outputs] == calls
    assert [row["name"] for row in outputs] == ["collect_1030 norm_stats", "data-1 norm_stats"]
    assert [row["metadata"]["source_dataset_name"] for row in outputs] == ["collect_1030", "data-1"]
    assert [row["path"] for row in outputs] == [str(output / "dataset-001/norm_stats.json"), str(output / "dataset-002/norm_stats.json")]
    assert [json.loads(Path(row["path"]).read_text())["norm_stats"]["state"]["mean"] for row in outputs] == [[0.], [1.]]
    assert not (output / "norm_stats.json").exists()


def test_same_named_batch_datasets_keep_their_own_source_ids_without_inventing_item_numbers(batch_job):
    spec, plan, calls, _ = batch_job
    for asset in spec["inputs"].values():
        asset["name"] = "采集数据😀"
    plan["generated_config"] = norm_entry.generated_config(spec)
    norm_entry.run_spec({**spec, "expected_plan_digest": canonical_digest(plan)})
    outputs = json.loads((Path(spec["output_dir"]) / "artifacts.json").read_text())["outputs"]
    assert [row["name"] for row in outputs] == ["采集数据😀 norm_stats"] * 2
    assert [row["metadata"]["source_dataset_asset_id"] for row in outputs] == calls == ["first", "second"]
    assert len({row["path"] for row in outputs}) == 2


@pytest.mark.parametrize("change", ["fail_second", "change_first"])
def test_batch_failure_or_source_drift_never_publishes_partial_asset_manifest(batch_job, change):
    spec, plan, _, state = batch_job
    state[change] = True
    with pytest.raises(ValueError, match="second dataset failed|DATASET_CHANGED"):
        norm_entry.run_spec({**spec, "expected_plan_digest": canonical_digest(plan)})
    output = Path(spec["output_dir"])
    assert (output / "dataset-001/norm_stats.json").is_file()
    assert not (output / "artifacts.json").exists()


def test_downloaded_batch_config_cannot_silently_compute_only_first_source(batch_job):
    spec, _, _, _ = batch_job
    source = norm_entry.generated_config(spec)["source"]
    ast.parse(source)
    before = list(sys.path)
    try:
        module = {}; exec(compile(source, "training_config.py", "exec"), module)
        with pytest.raises(ValueError, match="explicit dataset_index"):
            module["build_config"](spec["output_dir"])
        assert [config.dataset_id for config in module["build_configs"](spec["output_dir"])] == ["first", "second"]
    finally:
        sys.path[:] = before


def test_each_batch_row_selects_its_own_recipe_factory_or_processor_without_recipe_weights(tmp_path, monkeypatch):
    @dataclasses.dataclass
    class Assets:
        assets_dir: str
        asset_id: str

    @dataclasses.dataclass
    class ComposableDataConfig:
        repo_id: str
        assets: Assets
        scale: float = 1.
        composition: dict = dataclasses.field(default_factory=lambda: composed_profile()["processing"])

    @dataclasses.dataclass
    class Multi:
        datasets: list
        weights: object = None

    @dataclasses.dataclass
    class Config:
        data: object
        model: object = dataclasses.field(default_factory=lambda: SimpleNamespace(action_horizon=16))
        batch_size: int = 8
        num_workers: int = 4

    base = Config(Multi([ComposableDataConfig('/old/first', Assets('/old/norm', 'old'), 2.),
                         ComposableDataConfig('/old/second', Assets('/old/norm', 'old'), 7.)], weights=[100., 1.]))
    original = copy.deepcopy(base)
    module = ModuleType('openpi.training.config'); module.__file__ = str(tmp_path / 'src/openpi/training/config.py')
    module.AssetsConfig = Assets; module.get_config = lambda _: base
    training = ModuleType('openpi.training'); training.config = module
    openpi = ModuleType('openpi'); openpi.training = training
    for name, value in [('openpi', openpi), ('openpi.training', training), ('openpi.training.config', module)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(norm_entry, 'configure_cache', lambda _: None)
    monkeypatch.setattr(norm_entry, 'install_registered_fps', lambda _: None)
    inputs = {}
    for index in range(3):
        dataset = tmp_path / f'data-{index}'; (dataset / 'meta').mkdir(parents=True)
        (dataset / 'meta/info.json').write_text('{}')
        inputs[f'dataset_{index}'] = {'id':str(index), 'kind':'DATASET', 'path':str(dataset)}
    rows = [{'dataset_asset_id':'0', 'recipe_index':1}, {'dataset_asset_id':'1', 'recipe_index':0},
            {'dataset_asset_id':'2', 'processor':{'name':'ComposableDataConfig', 'parameters':{'composition':composed_profile()['processing']}}}]
    selected = []
    def processor(choice, config_module, code_dir, *, dataset_path, norm_dir, norm_stats=False):
        assert norm_stats
        selected.append((choice, str(dataset_path), str(norm_dir)))
        return ComposableDataConfig(str(dataset_path), Assets(str(norm_dir), '.'), 11.)
    monkeypatch.setattr(norm_entry, 'instantiate_data_processor', processor)
    spec = {'code_dir':str(tmp_path),'recipe':'multi','inputs':inputs,'norm_datasets':rows,
            'output_dir':str(tmp_path/'output'),'parameters':{'batch_size':32,'num_workers':0}}
    configs = [norm_entry.prepare_config({**spec,'norm_dataset_index':index}) for index in range(3)]
    assert [config.data.scale for config in configs] == [7.,2.,11.]
    assert [config.data.repo_id for config in configs] == [asset['path'] for asset in inputs.values()]
    assert all(config.data.assets.assets_dir == str(tmp_path/'output/.unbound') for config in configs)
    assert selected == [(normalize_norm_datasets(rows)[2]['processor'], inputs['dataset_2']['path'], str(tmp_path/'output/.unbound'))]
    assert base == original


def test_registered_fps_policy_covers_every_independent_statistics_row(tmp_path):
    from vla_adapters.dataset_fps import resolve_registered_fps
    inputs, rows = {}, []
    for index, fps in enumerate((20, 30)):
        dataset = tmp_path / f'data-{index}';(dataset/'meta').mkdir(parents=True)
        (dataset/'meta/info.json').write_text(json.dumps({'fps':fps}))
        rows.append({'dataset_asset_id':str(index),'recipe_index':0})
        inputs[f'dataset_{index}']={'id':str(index),'kind':'DATASET','path':str(dataset),'dataset_profile':{'fps':fps}}
    receipt = resolve_registered_fps({'inputs':inputs,'norm_datasets':rows,'allowed_roots':[str(tmp_path)]})
    assert len(receipt) == 2
    assert [row['effective_fps'] for row in receipt] == [20,30]
