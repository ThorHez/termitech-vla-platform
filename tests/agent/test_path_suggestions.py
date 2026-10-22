from datetime import datetime, timedelta, timezone

import pytest

from vla_agent.main import Agent
from vla_agent import path_suggestions
from vla_agent.path_suggestions import suggest_paths
from vla_agent.safety import AgentError


def query(prefix, **overrides):
    return {"prefix": prefix, "limit": 50,
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(), **overrides}


def test_path_completion_roots_children_and_file_names(tmp_path):
    root = tmp_path / "data2"
    root.mkdir()
    (root / "apple-dir").mkdir()
    (root / "apple.txt").write_text("contents must never be returned")
    (root / "banana").mkdir()
    (root / ".hidden").mkdir()
    config = {"roots": [str(root)]}
    expected_root = {"name": "data2", "path": f"{root}/", "kind": "directory"}
    for prefix in ("", "/", str(root.parent) + "/da", str(root)):
        result = suggest_paths(config, query(prefix))
        assert result == {"prefix": prefix, "parent": None, "entries": [expected_root], "truncated": False}
    result = suggest_paths(config, query(str(root) + "/a"))
    assert result["parent"] == str(root)
    assert result["entries"] == [
        {"name": "apple-dir", "path": f"{root}/apple-dir/", "kind": "directory"},
        {"name": "apple.txt", "path": f"{root}/apple.txt", "kind": "file"},
    ]
    assert "contents" not in str(result)
    assert [entry["name"] for entry in suggest_paths(config, query(str(root) + "/"))["entries"]] == ["apple-dir", "banana", "apple.txt"]
    assert suggest_paths(config, query(str(root) + "/A"))["entries"] == []
    assert [entry["name"] for entry in suggest_paths(config, query(str(root) + "/."))["entries"]] == [".hidden"]
    assert suggest_paths(config, query(f"{root}/apple-dir/"))["entries"] == []


def test_nested_roots_do_not_hide_other_parent_matches(tmp_path):
    (tmp_path / "openpi").mkdir()
    (tmp_path / "outputs").mkdir()
    result = suggest_paths({"roots": [str(tmp_path), str(tmp_path / "outputs")]}, query(f"{tmp_path}/o"))
    assert result["parent"] == str(tmp_path)
    assert [entry["name"] for entry in result["entries"]] == ["openpi", "outputs"]


def test_system_root_can_be_selected_then_opened(tmp_path, monkeypatch):
    (tmp_path / "directory").mkdir()
    (tmp_path / "model.bin").write_bytes(b"private model data")
    real_scandir = path_suggestions.os.scandir
    def root_listing(parent):
        assert str(parent) == "/"
        return real_scandir(tmp_path)
    monkeypatch.setattr(path_suggestions.os, "scandir", root_listing)
    config = {"roots": ["/"]}
    assert suggest_paths(config, query(""))["entries"] == [{"name": "/", "path": "/", "kind": "directory"}]
    result = suggest_paths(config, query("/"))
    assert result == {"prefix": "/", "parent": "/", "entries": [
        {"name": "directory", "path": "/directory/", "kind": "directory"},
        {"name": "model.bin", "path": "/model.bin", "kind": "file"},
    ], "truncated": False}


def test_symlink_root_keeps_configured_spelling_and_blocks_escape(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "dataset").mkdir()
    (real / "data.bin").write_bytes(b"private-content")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("not exposed")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    (real / "escaped").symlink_to(outside, target_is_directory=True)
    (real / "dangling").symlink_to(real / "absent")
    (real / "loop").symlink_to(real / "loop")
    (real / "inner-link").symlink_to(real / "dataset", target_is_directory=True)
    config = {"roots": [str(link)]}
    result = suggest_paths(config, query(f"{link}/"))
    assert [entry["name"] for entry in result["entries"]] == ["dataset", "inner-link", "data.bin"]
    assert all(entry["path"].startswith(f"{link}/") for entry in result["entries"])
    with pytest.raises(AgentError, match="resolves outside") as failure:
        suggest_paths(config, query(f"{link}/escaped/"))
    assert failure.value.code == "PATH_OUTSIDE_ROOTS"
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, query(f"{real}/"))
    assert failure.value.code == "PATH_OUTSIDE_ROOTS"


@pytest.mark.parametrize("prefix", ["relative/path", "/data/../etc", "/nul\x00", "/line\n", "/del\x7f", "/c1\x85", "/" + "x" * 4096, None, 4])
def test_invalid_prefix_rejected_before_reading_directories(tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(path_suggestions.os, "scandir", lambda *_: pytest.fail("must not scan invalid paths"))
    with pytest.raises(AgentError) as failure:
        suggest_paths({"roots": [str(tmp_path)]}, query(prefix))
    assert failure.value.code == "INVALID_PATH"


@pytest.mark.parametrize("limit", [True, False, 1.5, "2", None])
def test_invalid_limit_rejected(tmp_path, limit):
    with pytest.raises(AgentError) as failure:
        suggest_paths({"roots": [str(tmp_path)]}, query(str(tmp_path), limit=limit))
    assert failure.value.code == "INVALID_PATH"


def test_expired_query_never_resolves_or_scans(tmp_path, monkeypatch):
    monkeypatch.setattr(path_suggestions, "_accessible_root", lambda *_: pytest.fail("must not resolve expired paths"))
    monkeypatch.setattr(path_suggestions.os, "scandir", lambda *_: pytest.fail("must not scan expired paths"))
    with pytest.raises(AgentError) as failure:
        suggest_paths({"roots": [str(tmp_path)]}, query(str(tmp_path), expires_at="2000-01-01T00:00:00Z"))
    assert failure.value.code == "PATH_QUERY_EXPIRED"


@pytest.mark.parametrize("expiry", [None, "not-a-date", "2030-01-01T00:00:00", "2030-01-01T00:00:00+01:00"])
def test_invalid_expiry_rejected(tmp_path, expiry):
    with pytest.raises(AgentError) as failure:
        suggest_paths({"roots": [str(tmp_path)]}, query(str(tmp_path), expires_at=expiry))
    assert failure.value.code == "INVALID_PATH"


def test_missing_parent_and_permission_denied_are_distinct(tmp_path, monkeypatch):
    config = {"roots": [str(tmp_path)]}
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, query(f"{tmp_path}/absent/"))
    assert failure.value.code == "PATH_UNAVAILABLE"
    def denied(*_):
        raise PermissionError("private directory")
    monkeypatch.setattr(path_suggestions.os, "scandir", denied)
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, query(f"{tmp_path}/"))
    assert failure.value.code == "PATH_PERMISSION_DENIED"


def test_inaccessible_or_missing_roots_not_suggested(tmp_path, monkeypatch):
    monkeypatch.setattr(path_suggestions.os, "access", lambda *_: False)
    result = suggest_paths({"roots": [str(tmp_path), str(tmp_path / "missing")]}, query(""))
    assert result["entries"] == []


def test_only_plain_files_and_directories_suggested(tmp_path):
    import os
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "file").write_text("secret")
    (tmp_path / "bad\nname").write_text("secret")
    assert [entry["name"] for entry in suggest_paths({"roots": [str(tmp_path)]}, query(f"{tmp_path}/"))["entries"]] == ["file"]


def test_completion_never_opens_file_contents(tmp_path, monkeypatch):
    from pathlib import Path
    (tmp_path / "model.bin").write_bytes(b"private model data")
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("path completion must not open file contents"))
    result = suggest_paths({"roots": [str(tmp_path)]}, query(f"{tmp_path}/"))
    assert result["entries"] == [{"name": "model.bin", "path": f"{tmp_path}/model.bin", "kind": "file"}]


def test_large_directory_scan_and_result_limits_are_bounded(tmp_path, monkeypatch):
    for index in range(60):
        (tmp_path / f"entry-{index:03}").mkdir()
    config = {"roots": [str(tmp_path)]}
    result = suggest_paths(config, query(f"{tmp_path}/", limit=1000))
    assert len(result["entries"]) == 50 and result["truncated"]
    result = suggest_paths(config, query(f"{tmp_path}/", limit=0))
    assert len(result["entries"]) == 1 and result["truncated"]
    monkeypatch.setattr(path_suggestions, "MAX_SCANNED_ENTRIES", 10)
    result = suggest_paths(config, query(f"{tmp_path}/"))
    assert len(result["entries"]) == 10 and result["truncated"]


def test_scan_time_budget_returns_partial_results(tmp_path, monkeypatch):
    (tmp_path / "entry").mkdir()
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(path_suggestions.time, "monotonic", lambda: next(clock))
    result = suggest_paths({"roots": [str(tmp_path)]}, query(f"{tmp_path}/"))
    assert result["entries"] == [] and result["truncated"]


def test_path_command_persists_success_and_rejection_across_restarts(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    agent = Agent(config)
    command = {"id": "path-completion", "type": "PATH_SUGGEST", "payload": query(config["roots"][0] + "/")}
    acknowledgement = agent.handle_command(command)
    assert acknowledgement["status"] == "APPLIED"
    rejected = {"id": "path-rejected", "type": "PATH_SUGGEST", "payload": query("relative")}
    failure = agent.handle_command(rejected)
    assert failure["status"] == "REJECTED" and failure["result"]["error"]["code"] == "INVALID_PATH"
    restarted = Agent(config)
    monkeypatch.setattr("vla_agent.main.suggest_paths", lambda *_: pytest.fail("persisted commands must not rescan"))
    assert restarted.handle_command(command) == acknowledgement
    assert restarted.handle_command(rejected) == failure
    agent.runtime_monitor.close()
    restarted.runtime_monitor.close()


def project_query(config, prefix="", **overrides):
    profile = config["profiles"][0]
    return query(prefix, profile_id=profile["id"], base_path=profile["code_dir"], python_only=True, **overrides)


def test_project_completion_starts_at_project_and_only_offers_python_files(tmp_path):
    project = tmp_path / "openpi-umi"
    project.mkdir()
    (project / "scripts").mkdir()
    (project / "train.py").write_text("pass\n")
    (project / "config.json").write_text("{}")
    (project / "train.sh").write_text("exit 0")
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi", "code_dir": str(project)}]}
    for prefix in ("", str(project), str(project) + "/"):
        result = suggest_paths(config, project_query(config, prefix))
        assert result["prefix"] == prefix
        assert result["parent"] == str(project)
        assert result["profile_id"] == "openpi" and result["base_path"] == str(project)
        assert [entry["name"] for entry in result["entries"]] == ["scripts", "train.py"]


@pytest.mark.parametrize("prefix_kind", ["ancestor", "sibling", "lexical_escape"])
def test_project_completion_never_exposes_ancestor_or_other_allowed_projects(tmp_path, prefix_kind):
    project = tmp_path / "openpi-umi"
    project.mkdir()
    sibling = tmp_path / "other-project"
    sibling.mkdir()
    (sibling / "private.py").write_text("pass\n")
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi", "code_dir": str(project)}]}
    prefixes = {"ancestor": str(tmp_path) + "/", "sibling": str(sibling) + "/",
                "lexical_escape": str(project) + "/../other-project/"}
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, project_query(config, prefixes[prefix_kind]))
    assert failure.value.code in {"PATH_OUTSIDE_PROJECT", "INVALID_PATH"}


def test_project_completion_filters_links_to_sibling_project_and_rejects_opening_them(tmp_path):
    project = tmp_path / "openpi-umi"
    project.mkdir()
    sibling = tmp_path / "other-project"
    sibling.mkdir()
    (sibling / "private.py").write_text("pass\n")
    (project / "escaped").symlink_to(sibling, target_is_directory=True)
    (project / "escape.py").symlink_to(sibling / "private.py")
    (project / "scripts").mkdir()
    (project / "inside").symlink_to(project / "scripts", target_is_directory=True)
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi", "code_dir": str(project)}]}
    result = suggest_paths(config, project_query(config))
    assert [entry["name"] for entry in result["entries"]] == ["inside", "scripts"]
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, project_query(config, f"{project}/escaped/"))
    assert failure.value.code == "PATH_OUTSIDE_PROJECT"


def test_project_query_rejects_stale_runtime_selection_before_scanning(tmp_path, monkeypatch):
    project = tmp_path / "openpi-umi"
    project.mkdir()
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi", "code_dir": str(project)}]}
    payload = project_query(config)
    config["profiles"][0]["code_dir"] = str(tmp_path / "new-project")
    monkeypatch.setattr(path_suggestions.os, "scandir", lambda *_: pytest.fail("stale scope must not scan"))
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, payload)
    assert failure.value.code == "PATH_SCOPE_MISMATCH"
    payload["profile_id"] = "removed-profile"
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, payload)
    assert failure.value.code == "RUNTIME_UNAVAILABLE"


def test_project_query_keeps_symlink_project_spelling_and_checks_real_roots(tmp_path):
    project = tmp_path / "real-project"
    project.mkdir()
    (project / "train.py").write_text("pass\n")
    link = tmp_path / "openpi-umi"
    link.symlink_to(project, target_is_directory=True)
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi", "code_dir": str(link)}]}
    result = suggest_paths(config, project_query(config))
    assert result["entries"] == [{"name": "train.py", "path": str(link / "train.py"), "kind": "file"}]
    config["roots"] = [str(tmp_path / "different-root")]
    with pytest.raises(AgentError) as failure:
        suggest_paths(config, project_query(config))
    assert failure.value.code == "PATH_OUTSIDE_ROOTS"
