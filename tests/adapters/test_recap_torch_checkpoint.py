import json
from math import prod
from types import SimpleNamespace

import pytest

from vla_adapters.recap_checkpoint import recap_model_evidence_valid
from vla_adapters.training_plan import model_contract
from vla_agent.artifacts import discover_openpi_outputs


def _weights(path, shapes):
    header = {}
    offset = 0
    for name, shape in shapes.items():
        size = prod(shape) * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * offset)


@pytest.mark.parametrize("explicit_file", [False, True])
def test_recap_safetensors_checkpoint_is_bound_and_validated_without_torch(tmp_path, explicit_file):
    path = tmp_path / "model.safetensors"
    _weights(path, {"action_in_proj.weight": [3, 2], "action_out_proj.weight": [2, 3]})
    asset = str(path if explicit_file else tmp_path)
    spec = {"template_id": "recap", "recap_backend": "pytorch", "training_stage": "offline_rl",
            "inputs": {"model": {"path": asset}}, "allowed_roots": [str(tmp_path)]}
    model = SimpleNamespace(action_dim=2, pi05=True)
    evidence, _ = model_contract(spec, model)
    assert recap_model_evidence_valid(evidence, asset)
    assert evidence["path"] == str(path)

    model.action_dim = 4
    with pytest.raises(ValueError, match="action_dim"):
        model_contract(spec, model)
    model.action_dim = 2
    _weights(path, {"action_in_proj.weight": [3, 2], "action_out_proj.weight": [2, 3],
                    "value_head.0.weight": [3, 3]})
    with pytest.raises(ValueError, match="估计器类型"):
        model_contract(spec, model)


def test_discovery_requires_atomic_pytorch_commit_and_valid_payload(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "recipe" / "platform" / "19"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "model.safetensors"
    _weights(weights, {"value_head.0.weight": [3, 3]})
    (checkpoint / "assets").mkdir()
    assert discover_openpi_outputs(tmp_path, "recipe") == []

    marker = checkpoint / "recap_checkpoint.json"
    marker.write_text(json.dumps({"schema_version": 1, "backend": "pytorch", "stage": "stage_progress", "step": 19}))
    model, = discover_openpi_outputs(tmp_path, "recipe")
    assert model["metadata"]["format"] == "openpi-pytorch"
    assert model["metadata"]["model_role"] == "stage_progress_estimator"
    assert model["path"] == str(checkpoint)

    weights.write_bytes(weights.read_bytes()[:-1])
    assert discover_openpi_outputs(tmp_path, "recipe") == []
