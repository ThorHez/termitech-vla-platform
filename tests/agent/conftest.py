import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vla_agent.config import load_config
from vla_agent.snapshots import capture_code


@pytest.fixture
def agent_fixture(tmp_path):
    repo = tmp_path / "code"
    repo.mkdir()
    source = Path(__file__).resolve().parents[2] / "templates" / "smoke_training.py"
    shutil.copy(source, repo / "smoke_training.py")
    (repo / ".gitignore").write_text("__pycache__/\n")
    for args in (["init", "-b", "fixture"], ["config", "user.name", "Agent Tests"],
                 ["config", "user.email", "agent@example.invalid"], ["add", "."], ["commit", "-m", "Fixture source"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    config = {
        "control_plane_url": "http://127.0.0.1:1", "enrollment_token": "test-enrollment",
        "name": "test-agent", "roots": [str(tmp_path)], "output_root": str(tmp_path / "outputs"),
        "state_dir": str(tmp_path / "state"), "profiles": [{
            "id": "smoke-profile", "name": "Smoke", "code_dir": str(repo), "python": sys.executable,
            "adapter": "smoke", "execution_mode": "native", "metadata": {},
        }], "poll_interval_seconds": 0.1, "heartbeat_interval_seconds": 0.1, "cancel_grace_seconds": 0.2,
    }
    config_path = tmp_path / "agent.json"
    config_path.write_text(json.dumps(config))
    loaded = load_config(config_path)
    baseline = capture_code(loaded["profiles"][0], loaded["roots"])
    payload = {"run_id": "test-run", "attempt_id": "test-attempt", "execution_token": "test-token",
               "profile_id": "smoke-profile", "template_id": "smoke", "inputs": {},
               "parameters": {"steps": 4}, "gpu_uuids": [], "code_expectation": baseline}
    return {"config": loaded, "repo": repo, "payload": payload, "root": tmp_path}


def bind_config_confirmation(payload, automatic):
    """Explicitly confirm a synthetic OpenPI plan in tests that reach the runner."""
    from vla_protocol import canonical_digest
    source = "CONFIG = {\"fixture\": True}\n"
    generated = {"schema_version": 1, "language": "python", "filename": "training_config.py",
                 "source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}
    automatic["plan"]["generated_config"] = generated
    automatic["plan_digest"] = canonical_digest(automatic["plan"])
    payload["code_expectation"] = {**payload["code_expectation"], "automatic_preflight": automatic,
                                   "confirmed_config_sha256": generated["sha256"]}
    payload["confirmed_config_sha256"] = generated["sha256"]
    return generated
