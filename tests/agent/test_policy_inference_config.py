"""CPU-only generation witnesses the config actually handed to policy loading."""

from copy import deepcopy
from contextlib import nullcontext
import dataclasses
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_agent.policy_deployment_worker import configuration_receipt, probe, resolve_inference_config
from vla_adapters.portable_policy import export_portable_policy, portable_digest
from vla_adapters.policy_inference import ComposedInferenceInputs


@dataclasses.dataclass(frozen=True)
class ComposedInputs:
    composition: dict


@dataclasses.dataclass(frozen=True)
class Data:
    asset_id: str
    data_transforms: object
    model_transforms: object
    use_quantile_norm: bool = True
    normalize_masks: object = None
    robot_type: str = "ARX5"


@dataclasses.dataclass(frozen=True)
class Factory:
    value: Data

    def create(self, *_args):
        return self.value


@dataclasses.dataclass(frozen=True)
class Train:
    model: object
    data: object
    policy_metadata: dict
    assets_dirs: tuple = ()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    classes = {}
    for name in ("openpi.models.pi0_config", "openpi.transforms"):
        module = ModuleType(name)
        path = root / "src" / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned " + name + "\n")
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
    for name, module, fields in (
        ("Pi0Config", "openpi.models.pi0_config", [("action_dim", int, 32), ("action_horizon", int, 77),
            ("image_keys", tuple, ("left_wrist_0_rgb",))]),
        ("Group", "openpi.transforms", [("inputs", tuple, ()), ("outputs", tuple, ())]),
        ("Scale", "openpi.transforms", [("factor", float, 2.5)]),
    ):
        cls = dataclasses.make_dataclass(name, fields, frozen=True)
        cls.__module__ = module
        setattr(sys.modules[module], name, cls)
        classes[name] = cls
    spec = json.loads((Path(__file__).parents[1] / "fixtures/composed-processing-v1.json").read_text())["valid"][0]
    group, scale = classes["Group"], classes["Scale"]
    data = Data("norm-14", group((ComposedInputs(spec),), (scale(3.5),)),
                group((scale(),), (scale(4.5),)),
                normalize_masks={"state": (True,) * 6 + (False,), "actions": (True,) * 3 + (False,) * 7})
    train = Train(classes["Pi0Config"](), Factory(data), {"task": "fold", "layout": "wrist-chw"})
    directory = tmp_path / "policy-deployments" / "deployment"
    directory.mkdir(parents=True)
    manifest = {"code_dir": str(root), "config_name": "base_recipe", "source_run_id": "run-training",
                "generated_sha256": "a" * 64, "frozen_request": {"recipe": "base_recipe", "parameters": {"horizon": 77}},
                "deployment_id": "deployment", "state_dir": str(tmp_path), "inference_config_schema": 1}
    prepared = []
    module = ModuleType("vla_adapters.openpi_entry")
    def prepare_config(spec):
        prepared.append(deepcopy(spec))
        return train, {}
    module.prepare_config = prepare_config
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("vla_adapters.generated_config.generated_training_config",
                        lambda *_args: {"sha256": manifest["generated_sha256"]})
    def forbidden_lookup(_name):
        raise AssertionError("Training config must never fall back to its base recipe")
    return root, directory, manifest, train, prepared, SimpleNamespace(get_config=forbidden_lookup)


def test_frozen_training_generates_actual_inference_artifact_without_gpu(runtime, monkeypatch):
    root, directory, manifest, original, prepared, registry = runtime
    monkeypatch.setattr("vla_agent.policy_deployment_worker.check_gpu_before_load",
                        lambda *_args: pytest.fail("Generating configuration must not query GPU state"))
    restored, receipt = resolve_inference_config(manifest, directory, registry)
    assert prepared == [{**manifest["frozen_request"], "output_dir": str(directory / "configuration")}]
    assert restored.model == original.model and restored.model.action_horizon == 77
    data = restored.data.create(None, None)
    assert data.asset_id == "norm-14" and data.use_quantile_norm is True
    assert data.normalize_masks == original.data.value.normalize_masks
    assert data.model_transforms.inputs == list(original.data.value.model_transforms.inputs)
    assert data.model_transforms.outputs[:-1] == list(original.data.value.model_transforms.outputs)
    assert data.data_transforms.outputs == original.data.value.data_transforms.outputs
    transform = data.data_transforms.inputs[0]
    assert isinstance(transform, ComposedInferenceInputs)
    assert transform.composition["cameras"][0]["layout"] == "chw"
    assert transform.composition["action_layout"] == original.data.value.data_transforms.inputs[0].composition["action_layout"]
    artifact = json.loads((directory / "inference-config.json").read_text())
    assert receipt == {"source": "training", "config_name": "base_recipe", "source_run_id": "run-training",
                       "format": "portable-json",
                       "generated_sha256": "a" * 64, "digest": portable_digest(artifact["config"]),
                       "path": str(directory / "inference-config.json")}
    assert artifact["digest"] == receipt["digest"]
    assert artifact["config"]["model"]["fields"]["action_horizon"] == 77
    assert "ComposedInputs" not in json.dumps(artifact)
    assert restored.policy_metadata == original.policy_metadata
    assert (directory / "inference-config.json").stat().st_mode & 0o777 == 0o600


def test_changed_training_config_fails_without_recipe_fallback_or_artifact(runtime, monkeypatch):
    _, directory, manifest, _, _, registry = runtime
    monkeypatch.setattr("vla_adapters.generated_config.generated_training_config", lambda *_args: {"sha256": "b" * 64})
    with pytest.raises(ValueError, match="CONFIG_CHANGED"):
        resolve_inference_config(manifest, directory, registry)
    assert not (directory / "inference-config.json").exists()


def test_local_custom_training_transform_keeps_frozen_config_and_marks_resolver_dependency(runtime):
    _, directory, manifest, original, prepared, registry = runtime
    @dataclasses.dataclass(frozen=True)
    class CustomTransform:
        joint_offset: float = 1.25
    group = original.data.value.model_transforms
    custom = CustomTransform()
    object.__setattr__(original.data, "value", dataclasses.replace(original.data.value,
        model_transforms=dataclasses.replace(group, inputs=(*group.inputs, custom))))
    restored, receipt = resolve_inference_config(manifest, directory, registry)
    assert prepared and receipt["format"] == "local-frozen-resolver"
    assert restored.model == original.model and restored.model.action_horizon == 77
    assert restored.data.create(None, None).model_transforms.inputs[-1] is custom
    assert restored.data.create(None, None).normalize_masks == original.data.value.normalize_masks
    artifact = json.loads((directory / "inference-config.json").read_text())
    assert artifact["config"]["portable"] is False
    assert artifact["config"]["resolver"] == "checked-frozen-training-request"
    assert artifact["config"]["frozen_request"] == manifest["frozen_request"]
    assert "CustomTransform" in json.dumps(artifact) and "joint_offset" in json.dumps(artifact)


def test_codec_integrity_or_contract_failure_never_uses_local_resolver_fallback(runtime, monkeypatch):
    _, directory, manifest, _, _, registry = runtime
    def damaged_export(*_args, **_kwargs):
        raise ValueError("PORTABLE_POLICY_UNSUPPORTED: normalization contract is invalid")
    monkeypatch.setattr("vla_adapters.portable_policy.export_portable_policy", damaged_export)
    with pytest.raises(ValueError, match="normalization contract"):
        resolve_inference_config(manifest, directory, registry)
    assert not (directory / "inference-config.json").exists()


def test_portable_artifact_keeps_transfer_config_and_its_provenance(runtime):
    root, directory, manifest, original, prepared, registry = runtime
    portable = export_portable_policy(original, root)
    manifest.pop("frozen_request")
    manifest.pop("generated_sha256")
    manifest.update(portable_policy={"config": portable, "digest": portable_digest(portable)},
                    transfer_id="transfer-1", source_node_id="source-node")
    restored, receipt = resolve_inference_config(manifest, directory, registry)
    assert not prepared
    artifact = json.loads((directory / "inference-config.json").read_text())
    assert artifact["config"] == portable and artifact["digest"] == manifest["portable_policy"]["digest"]
    assert receipt["source"] == "portable" and receipt["transfer_id"] == "transfer-1"
    assert receipt["source_run_id"] == "run-training" and receipt["source_node_id"] == "source-node"
    assert restored.model.action_horizon == 77
    artifact["config"]["model"]["fields"]["action_horizon"] = 1
    (directory / "inference-config.json").write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="digest differs"):
        configuration_receipt(directory)


def test_manual_custom_recipe_still_uses_registry_without_portable_restrictions(runtime):
    _, directory, manifest, original, _, _ = runtime
    manifest.pop("frozen_request")
    custom = dataclasses.replace(original, model=object())
    names = []
    def get_config(name):
        names.append(name)
        return custom
    restored, receipt = resolve_inference_config(manifest, directory, SimpleNamespace(get_config=get_config))
    assert names == ["base_recipe"] and restored.model is custom.model
    assert restored.data.create(None, None).normalize_masks == custom.data.value.normalize_masks
    assert receipt is None and not (directory / "inference-config.json").exists()


@pytest.mark.parametrize("legacy_source", ("frozen", "portable"))
@pytest.mark.parametrize("execution_id", (None, "execution-restarted"))
def test_health_matches_generated_config_receipt_and_keeps_legacy_services_compatible(runtime, monkeypatch, legacy_source, execution_id):
    _, directory, manifest, _, _, registry = runtime
    if execution_id:
        execution_directory = directory.with_name(execution_id)
        directory.rename(execution_directory)
        directory = execution_directory
        manifest["execution_id"] = execution_id
    _, receipt = resolve_inference_config(manifest, directory, registry)
    manifest.update(port=12345, nonce="deployment-secret")
    metadata = {"_termitech_deployment": manifest["nonce"], "_termitech_inference_config": receipt}
    websocket, sync, client = (ModuleType(name) for name in ("websockets", "websockets.sync", "websockets.sync.client"))
    client.connect = lambda *_args, **_kwargs: nullcontext(SimpleNamespace(recv=lambda **_kwargs: b"metadata"))
    websocket.sync, sync.client = sync, client
    openpi_client = ModuleType("openpi_client")
    openpi_client.msgpack_numpy = SimpleNamespace(unpackb=lambda _data: metadata)
    for module in (websocket, sync, client, openpi_client):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    assert probe(manifest) == {"ok": True}
    manifest["resolved_backend"] = "pytorch"
    with pytest.raises(ValueError, match="POLICY_BACKEND_MISMATCH"):
        probe(manifest)
    metadata["_termitech_inference_backend"] = "jax"
    with pytest.raises(ValueError, match="POLICY_BACKEND_MISMATCH"):
        probe(manifest)
    metadata["_termitech_inference_backend"] = "pytorch"
    assert probe(manifest) == {"ok": True}
    manifest.pop("resolved_backend")
    metadata["_termitech_inference_config"] = {**receipt, "digest": "b" * 64}
    with pytest.raises(ValueError, match="generated configuration does not match"):
        probe(manifest)
    metadata["_termitech_inference_config"] = receipt
    (directory / "inference-config.json").unlink()
    with pytest.raises(OSError):
        probe(manifest)
    # Already running workers from older Agent versions have no artifact claim.
    manifest.pop("inference_config_schema")
    if legacy_source == "portable":
        manifest.pop("frozen_request")
        manifest["portable_policy"] = {"config": {}, "digest": "a" * 64}
    metadata.pop("_termitech_inference_config")
    assert probe(manifest) == {"ok": True}
