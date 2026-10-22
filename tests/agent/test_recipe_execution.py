import hashlib

import pytest

from vla_agent.recipe_execution import verify_recipe_source
from vla_agent.safety import AgentError
from vla_adapters.recipe_definitions import definition_digest


def test_saved_recipe_requires_exact_source_and_tracked_code_version(tmp_path):
    path = tmp_path / "src/openpi/training/config.py"
    path.parent.mkdir(parents=True)
    path.write_text("_CONFIGS = []\n")
    source = {"path": "src/openpi/training/config.py", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "line": 1, "end_line": 1, "commit_sha": "a" * 40, "changes_digest": "b" * 64,
              "code_dir": str(tmp_path), "dirty": False}
    saved = {"schema_version": 1, "id": "revision", "recipe_id": "recipe", "name": "clone", "revision": 1,
             "node_id": "node", "profile_id": "profile", "base_recipe": "base", "source": source, "overrides": {}}
    saved["digest"] = definition_digest(saved)
    profile = {"id": "profile", "code_dir": str(tmp_path)}
    snapshot = {"commit_sha": source["commit_sha"], "changes_digest": source["changes_digest"]}
    verify_recipe_source(profile, [str(tmp_path)], saved, snapshot)
    with pytest.raises(AgentError, match="基础代码版本"):
        verify_recipe_source(profile, [str(tmp_path)], saved, {**snapshot, "changes_digest": "c" * 64})
    path.write_text("_CONFIGS = [changed]\n")
    with pytest.raises(AgentError, match="基础配方源码"):
        verify_recipe_source(profile, [str(tmp_path)], saved, snapshot)
