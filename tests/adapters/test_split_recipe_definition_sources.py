"""Saved split-file definitions are data-only and keep exact source pinning."""

import hashlib

import pytest

from vla_adapters.recipe_definitions import RECIPE_PART_PATHS, definition_digest, validate_definition
from vla_agent.recipe_execution import verify_recipe_source
from vla_agent.safety import AgentError


def definition(root, path, contents):
    value = {"schema_version": 1, "id": "revision", "recipe_id": "recipe", "name": "split recipe", "revision": 1,
             "node_id": "node", "profile_id": "runtime", "base_recipe": "base", "overrides": {},
             "source": {"path": path, "code_dir": str(root), "sha256": hashlib.sha256(contents).hexdigest(),
                        "line": 1, "end_line": 1, "commit_sha": "a" * 40, "changes_digest": "b" * 64, "dirty": False}}
    value["digest"] = definition_digest(value)
    return value


@pytest.mark.parametrize("path", sorted(RECIPE_PART_PATHS))
def test_saved_split_source_is_verified_and_later_changes_fail(tmp_path, path):
    contents = b"_CONFIGS = [TrainConfig(name='base')]\n"
    source = tmp_path / path
    source.parent.mkdir(parents=True)
    source.write_bytes(contents)
    saved = definition(tmp_path, path, contents)
    assert validate_definition(saved) == saved
    profile = {"id": "runtime", "code_dir": str(tmp_path)}
    snapshot = {"commit_sha": "a" * 40, "changes_digest": "b" * 64}
    verify_recipe_source(profile, [str(tmp_path)], saved, snapshot)
    source.write_bytes(contents + b"# changed\n")
    with pytest.raises(AgentError, match="基础配方源码"):
        verify_recipe_source(profile, [str(tmp_path)], saved, snapshot)


@pytest.mark.parametrize("path", ["../recipes/sft.py", "src/openpi/training/recipes/other.py",
    "src/openpi/training/recipes/__init__.py", "/etc/passwd"])
def test_a_known_recipe_shape_cannot_authorize_arbitrary_source_files(tmp_path, path):
    with pytest.raises(ValueError, match="invalid source path"):
        validate_definition(definition(tmp_path, path, b"_CONFIGS=[]\n"))
