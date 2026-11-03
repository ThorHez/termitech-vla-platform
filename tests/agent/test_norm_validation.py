import json
from pathlib import Path

import pytest

from vla_adapters.norm_entry import generated_config
from vla_agent import runner, training_validation
from vla_agent.safety import AgentError
from vla_protocol import canonical_digest
from vla_protocol.norm_plan import NORM_CHECKS


def test_norm_preflight_checks_numeric_projection_and_storage_without_hf_builders_or_nvml(agent_fixture, monkeypatch):
    root = agent_fixture["root"]
    dataset = root / "data"
    dataset.mkdir()
    script = agent_fixture["repo"] / "scripts/compute_norm_stats.py"
    script.parent.mkdir()
    script.write_text("# statistics entrypoint\n")
    spec = {"template_id": "openpi-norm", "code_dir": str(agent_fixture["repo"]), "recipe": "recipe",
            "inputs": {"dataset": {"kind": "DATASET", "path": str(dataset)}}, "parameters": {"batch_size": 32, "num_workers": 0},
            "gpu_count": 0, "output_dir": str(Path(agent_fixture["config"]["output_root"]) / "unused")}
    cache = str(root / "cache/datasets")
    columns = ["observation.state", "action"]
    plan = {"template_id": "openpi-norm", "recipe": "recipe", "recipe_definition_digest": None,
            "input_authority": "registered_assets", "requested_inputs": {"dataset": str(dataset)},
            "inputs": {"dataset": str(dataset)}, "parameters": spec["parameters"], "gpu_count": 0,
            "entrypoint": str(script), "entrypoint_sha256": "a" * 64, "adapter_sha256": "b" * 64,
            "generated_config": generated_config(spec),
            "datasets": [{"path": str(dataset), "frames": 32, "metadata_sha256": "c" * 64,
                          "manifest_sha256": "d" * 64, "sample_shapes": {"state": [14], "actions": [50, 14]},
                          "actions_shape": [50, 14], "numeric_io": {"schema_version": 1,
                              "reader": "pyarrow_parquet_projection", "columns": columns,
                              "state_columns": ["observation.state"], "action_column": "action",
                              "temporal_columns": [], "action_sequence": "packed", "image_columns": [], "video_decode": False,
                              "source": {"path": "src/openpi/training/numeric_stats.py", "sha256": "e" * 64}}}],
            "training_cache": {"mode": "numeric_projection", "persistent_cache": False,
                "paths": {"HF_DATASETS_CACHE": cache}, "datasets": [{"path": str(dataset), "columns": columns}]},
            "storage": {"required_bytes": 64, "reference_checkpoint_bytes": 0, "model_params_bytes": 0}}
    response = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
                "checks": [{"name": name, "ok": True, "message": "checked"} for name in NORM_CHECKS]}
    monkeypatch.setattr(training_validation, "_probe_process", lambda *args, **kwargs:
                        (0, "VLA_PREFLIGHT:" + json.dumps(response), ""))
    monkeypatch.setattr(training_validation, "gpu_inventory", lambda: pytest.fail("CPU statistics must not query NVML"))
    monkeypatch.setattr(training_validation, "_validate_training_cache", lambda *args:
                        pytest.fail("Statistics must not require Hugging Face dataset builders"))
    launch = {"argv": ["python", "norm_entry.py", "--spec-json", json.dumps(spec)], "cwd": spec["code_dir"], "env": {}}
    checks = []
    evidence = training_validation.validate_training(agent_fixture["config"], launch, checks, assigned=[])
    assert evidence["plan_digest"] == canonical_digest(plan)
    assert not Path(spec["output_dir"]).exists()
    assert any(row["name"] == "output_storage" and row["ok"] for row in checks)
    configured = {**agent_fixture["config"], "cache_root": str(root / "configured-cache")}
    with pytest.raises(AgentError, match="statistics cache differs from configured cache root"):
        training_validation.validate_training(configured, launch, [])
    response["plan"]["inputs"]["dataset"] = str(root / "other-data")
    with pytest.raises(AgentError, match="Missing dataset identity"):
        training_validation.validate_training(agent_fixture["config"], launch, [])


def test_numeric_cache_does_not_weaken_training_cache_or_allow_source_symlinks(tmp_path):
    root = tmp_path / "managed"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    source = root / "dataset"
    source.mkdir()
    columns = ["state", "actions"]
    plan = {"datasets": [{"path": str(source), "numeric_io": {"columns": columns}}],
            "training_cache": {"mode": "numeric_projection", "persistent_cache": False,
                "paths": {}, "datasets": [{"path": str(source), "columns": columns}]}}
    training_validation._validate_numeric_cache(plan, [str(root)])
    with pytest.raises(ValueError):
        training_validation._validate_training_cache(plan, [str(root)])
    alias = root / "alias"
    alias.symlink_to(external, target_is_directory=True)
    plan["training_cache"]["paths"]["TMPDIR"] = str(alias)
    with pytest.raises(ValueError, match="outside managed roots"):
        training_validation._validate_numeric_cache(plan, [str(root)])


def test_optional_dataset_name_survives_agent_path_resolution_and_statistics_launch(agent_fixture, monkeypatch):
    from vla_agent import preflight
    from vla_adapters import build_launch
    root = agent_fixture["root"]
    dataset = root / "collect_1030"; dataset.mkdir()
    alias = root / "dataset-alias"; alias.symlink_to(dataset, target_is_directory=True)
    config = agent_fixture["config"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    rows = [{"dataset_asset_id": "source-dataset", "recipe_index": 0}]
    asset = {"id": "source-dataset", "kind": "DATASET", "path": str(alias) + "/", "name": "采集数据 1030"}
    payload = {"template_id": "openpi-norm", "profile_id": config["profiles"][0]["id"],
               "norm_datasets": rows, "inputs": {"dataset_0": asset}}
    monkeypatch.setattr(preflight, "validate_runtime", lambda *args: None)
    profile, inputs = preflight.validate_local_inputs(config, payload)
    assert inputs["dataset_0"]["path"] == str(dataset.resolve())
    assert inputs["dataset_0"]["name"] == asset["name"]
    launch = build_launch(profile, "openpi-norm", inputs, {}, str(root / "output"), [], recipe="recipe", norm_datasets=rows)
    prepared = training_validation.prepare_launch(launch, config, payload["inputs"])
    frozen = json.loads(prepared["argv"][prepared["argv"].index("--spec-json") + 1])
    assert frozen["inputs"]["dataset_0"]["name"] == asset["name"]
    assert frozen["inputs"]["dataset_0"]["id"] == "source-dataset"
    assert frozen["requested_inputs"]["dataset_0"] == str(alias) + "/"


def test_runner_accepts_only_valid_norm_output_for_statistics_task(tmp_path):
    norm = tmp_path / "norm_stats.json"
    norm.write_text(json.dumps({"norm_stats": {key: {"mean": [0.], "std": [1.], "q01": [-1.], "q99": [1.]}
                               for key in ("state", "actions")}}))
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(json.dumps({"outputs": [{"kind": "NORM", "name": "Stats", "path": str(norm), "version": "1"}]}))
    assert runner.verified_outputs(tmp_path, False, template_id="openpi-norm")[0]["kind"] == "NORM"
    with pytest.raises(AgentError, match="MODEL"):
        runner.verified_outputs(tmp_path, False)
    norm.write_text('{"norm_stats": {}}')
    with pytest.raises(AgentError, match="ARTIFACT|Cannot verify"):
        runner.verified_outputs(tmp_path, False, template_id="openpi-norm")


def test_batch_provenance_binds_every_ordered_dataset_and_processing_choice(agent_fixture):
    from vla_agent.snapshots import capture_code, compare_code
    profile = agent_fixture['config']['profiles'][0]
    rows = [{'dataset_asset_id':'first','recipe_index':0},{'dataset_asset_id':'second','recipe_index':1}]
    first = capture_code(profile, agent_fixture['config']['roots'], recipe='smoke', norm_datasets=rows)
    assert first['norm_datasets'] == rows
    same = capture_code(profile, agent_fixture['config']['roots'], recipe='smoke', norm_datasets=rows)
    compare_code(first,same)
    changed = capture_code(profile, agent_fixture['config']['roots'], recipe='smoke', norm_datasets=list(reversed(rows)))
    with pytest.raises(AgentError,match='Code changed'):
        compare_code(first,changed)


def test_runner_rejects_missing_or_duplicated_batch_norm_outputs(tmp_path):
    rows = [{'dataset_asset_id':'first','recipe_index':0},{'dataset_asset_id':'second','recipe_index':0}]
    outputs = []
    for index,row in enumerate(rows):
        path=tmp_path/f'norm-{index}.json'
        path.write_text(json.dumps({'norm_stats':{key:{'mean':[0.],'std':[1.],'q01':[-1.],'q99':[1.]}
            for key in ('state','actions')}}))
        outputs.append({'kind':'NORM','name':str(index),'path':str(path),'version':'1',
                        'metadata':{'source_dataset_asset_id':row['dataset_asset_id']}})
    manifest=tmp_path/'artifacts.json';manifest.write_text(json.dumps({'outputs':outputs}))
    assert len(runner.verified_outputs(tmp_path,False,template_id='openpi-norm',norm_datasets=rows)) == 2
    for bad in (outputs[:1], [outputs[0],outputs[0]], list(reversed(outputs))):
        manifest.write_text(json.dumps({'outputs':bad}))
        with pytest.raises(AgentError,match='every independent input dataset'):
            runner.verified_outputs(tmp_path,False,template_id='openpi-norm',norm_datasets=rows)


def test_norm_automatic_snapshot_freezes_model_choices_and_every_processor(agent_fixture, monkeypatch):
    from types import SimpleNamespace
    from vla_agent import snapshots
    from vla_adapters.automatic_config import normalize_auto_config
    profile = {**agent_fixture["config"]["profiles"][0], "adapter": "openpi-sft"}
    source = agent_fixture["repo"] / "openpi.py"
    source.write_text("# registered OpenPI origin\n")
    execute = snapshots.subprocess.run

    def probe(argv, *args, **kwargs):
        if argv[0] == profile["python"] and "-c" in argv:
            return SimpleNamespace(stdout=json.dumps({"origin": str(source)}))
        return execute(argv, *args, **kwargs)

    monkeypatch.setattr(snapshots.subprocess, "run", probe)
    rows = [{"dataset_asset_id": "first", "processor": {"name": "IndependentFactory", "parameters": {}}}]
    auto = normalize_auto_config({"action_horizon": 16})
    first = snapshots.capture_code(profile, agent_fixture["config"]["roots"], auto_config=auto, norm_datasets=rows)
    assert first["recipe"] == "auto_pi05_full_h16_d32_sft"
    assert first["auto_config"] == auto and first["norm_datasets"] == rows
    changed = snapshots.capture_code(profile, agent_fixture["config"]["roots"],
                                     auto_config={**auto, "action_horizon": 32}, norm_datasets=rows)
    with pytest.raises(AgentError, match="Code changed"):
        snapshots.compare_code(first, changed)
