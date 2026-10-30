"""Exact GPU assignment contracts, exercised without GPU access or VLA training."""

import json
from pathlib import Path

import pytest

from vla_adapters import build_launch
from vla_agent import runner
from vla_agent.metrics import check_gpus
from vla_agent.safety import AgentError
from vla_agent.snapshots import capture_code
from vla_agent.state import State


def test_launch_preserves_assigned_uuid_order_and_count(tmp_path):
    profile = {"adapter": "openpi-sft", "code_dir": str(tmp_path),
               "python": "/usr/bin/python3", "recipe": "fixture_recipe"}
    inputs = {role: {"path": str(tmp_path / role)} for role in ("dataset", "model", "norm")}
    assigned = ["GPU-z-selected", "GPU-a-selected"]
    launch = build_launch(profile, "openpi-sft", inputs, {"batch_size": 8},
                          str(tmp_path / "output"), assigned)
    assert launch["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-z-selected,GPU-a-selected"
    assert json.loads(launch["argv"][-1])["gpu_count"] == 2
    assert assigned == ["GPU-z-selected", "GPU-a-selected"]


def test_runner_child_receives_only_assigned_gpus_in_original_order(agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    assigned = ["GPU-z-selected", "GPU-a-selected"]
    payload["gpu_uuids"] = assigned.copy()
    payload["parameters"]["steps"] = 1
    script = agent_fixture["repo"] / "smoke_training.py"
    script.write_text('''import argparse,json,os
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',required=True)
args,_=parser.parse_known_args()
Path(args.output_dir,'child-gpu-env.json').write_text(json.dumps(os.environ.get('CUDA_VISIBLE_DEVICES')))
''')
    payload["code_expectation"] = capture_code(config["profiles"][0], config["roots"])
    state = State(config["state_dir"])
    output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    checked = []
    monkeypatch.setattr(runner, "check_gpus", lambda values: checked.append(list(values)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-unrelated-host-device")

    assert runner.execute(config, payload["attempt_id"]) == 0
    assert state.attempt(payload["attempt_id"])["state"] == "SUCCEEDED"
    assert json.loads((output / "child-gpu-env.json").read_text()) == ",".join(assigned)
    assert checked == [assigned]
    assert payload["gpu_uuids"] == assigned


@pytest.mark.parametrize("selected, expected_code", [
    ({"uuid": "GPU-selected", "processes": [{"pid": 123}]}, "GPU_BUSY"),
    ({"uuid": "GPU-selected", "processes": None}, "GPU_UNAVAILABLE"),
    (None, "GPU_UNAVAILABLE"),
])
def test_busy_or_missing_selected_gpu_never_falls_back_to_another_free_gpu(selected, expected_code):
    inventory = [{"uuid": "GPU-other-free", "processes": []}]
    if selected is not None:
        inventory.append(selected)
    with pytest.raises(AgentError) as error:
        check_gpus(["GPU-selected"], inventory=(inventory, {"available": True}))
    assert error.value.code == expected_code
