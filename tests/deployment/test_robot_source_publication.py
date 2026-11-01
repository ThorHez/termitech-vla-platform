"""Published robot snapshots retain their patch provenance without running source."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("robot_sources_publication_fixture", ROOT / "scripts/robot_sources.py")
sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sources)

BASE_SOURCE = b"raise RuntimeError('fixture source must never execute')\n# base\n"
PATCHED_SOURCE = b"raise RuntimeError('fixture source must never execute')\n# patched\n"
DEPLOY_SOURCE = b"#!/bin/sh\nexit 99\n"
PATCH = b"""diff --git a/source.py b/source.py
--- a/source.py
+++ b/source.py
@@ -1,2 +1,2 @@
 raise RuntimeError('fixture source must never execute')
-# base
+# patched
diff --git a/obsolete.txt b/obsolete.txt
deleted file mode 100644
--- a/obsolete.txt
+++ /dev/null
@@ -1 +0,0 @@
-obsolete
diff --git a/deploy.sh b/deploy.sh
new file mode 100755
--- /dev/null
+++ b/deploy.sh
@@ -0,0 +1,2 @@
+#!/bin/sh
+exit 99
"""


def git(project, *args):
    return subprocess.run(
        ["git", "-C", str(project), *args], check=True, capture_output=True,
    ).stdout.decode().strip()


def commit(project, message):
    git(project, "add", "--all")
    git(project, "-c", "user.name=Source fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", message)
    return git(project, "rev-parse", "HEAD")


def identity(path, content, mode):
    return {"path": path, "size": len(content), "mode": mode, "sha256": hashlib.sha256(content).hexdigest()}


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path
    patch = root / "third_party/patches/arx5-sdk-deployment.patch"
    patch.parent.mkdir(parents=True)
    patch.write_bytes(PATCH)
    project = root / sources.PROJECT_PATHS["arx5-sdk"]
    project.mkdir()
    git(project, "init", "-q")
    (project / "source.py").write_bytes(BASE_SOURCE)
    (project / "obsolete.txt").write_bytes(b"obsolete\n")
    base = commit(project, "baseline fixture")
    item = {
        "name": "arx5-sdk", "path": sources.PROJECT_PATHS["arx5-sdk"], "base_commit": base,
        "patch": {"path": patch.relative_to(root).as_posix(), "sha256": hashlib.sha256(PATCH).hexdigest()},
        "files": [identity("source.py", PATCHED_SOURCE, "100644"), identity("deploy.sh", DEPLOY_SOURCE, "100755")],
    }
    # The selected-project API still requires both project names in the schema;
    # the unselected entry is never initialized or accessed by these tests.
    manifest = {"schema_version": 1, "projects": [item, {"name": "umi-arx-deploy"}]}

    def save():
        (root / sources.MANIFEST).write_text(json.dumps(manifest))

    save()
    return root, project, patch, item, save


def publish(snapshot):
    _, project, patch, item, save = snapshot
    git(project, "apply", str(patch))
    item["published_commit"] = commit(project, "publish locked source fixture")
    save()
    return item["published_commit"]


def test_base_and_patch_mode_remains_compatible_without_publication(snapshot):
    root, project, _, item, _ = snapshot
    row = sources.run(root, "check", "arx5-sdk")["projects"][0]
    assert row["status"] == "patch_required"
    assert row["head_commit"] == item["base_commit"]
    assert row["published_commit"] is None
    assert (project / "source.py").read_bytes() == BASE_SOURCE
    row = sources.run(root, "apply", "arx5-sdk")["projects"][0]
    assert row["status"] == "ready"
    assert row["verified_files"] == 2
    assert not (project / "obsolete.txt").exists()
    assert sources.run(root, "check", "arx5-sdk")["projects"][0]["status"] == "ready"


def test_published_clean_snapshot_check_and_apply_are_idempotent(snapshot):
    root, project, _, item, _ = snapshot
    published = publish(snapshot)
    for operation in ("check", "apply", "apply", "check"):
        row = sources.run(root, operation, "arx5-sdk")["projects"][0]
        assert row["status"] == "ready"
        assert row["head_commit"] == row["published_commit"] == published
        assert row["base_commit"] == item["base_commit"]
        assert row["verified_files"] == 2
        assert git(project, "status", "--porcelain") == ""
        assert git(project, "rev-parse", "HEAD") == published


def test_unpinned_head_is_rejected_without_changes(snapshot):
    root, project, _, _, _ = snapshot
    publish(snapshot)
    (project / "extra.txt").write_text("unapproved commit\n")
    wrong = commit(project, "unapproved fixture")
    with pytest.raises(sources.SourceError, match="HEAD differs"):
        sources.run(root, "apply", "arx5-sdk")
    assert git(project, "rev-parse", "HEAD") == wrong
    assert git(project, "status", "--porcelain") == ""


@pytest.mark.parametrize("invalid", [None, "", "a" * 39, "a" * 41, "A" * 40, "g" * 40, 123])
def test_invalid_published_commit_identity_is_rejected(snapshot, invalid):
    root, project, _, item, save = snapshot
    item["published_commit"] = invalid
    save()
    with pytest.raises(sources.SourceError, match="published commit must be a full lowercase Git SHA"):
        sources.run(root, "check", "arx5-sdk")
    assert (project / "source.py").read_bytes() == BASE_SOURCE


@pytest.mark.parametrize("drift", ["content", "deleted_file", "mode"])
def test_published_snapshot_drift_is_rejected_without_repair(snapshot, drift):
    root, project, _, _, _ = snapshot
    published = publish(snapshot)
    if drift == "content":
        (project / "source.py").write_bytes(b"local work\n")
    elif drift == "deleted_file":
        (project / "obsolete.txt").write_bytes(b"obsolete\n")
    else:
        (project / "deploy.sh").chmod(0o644)
    before = git(project, "status", "--porcelain")
    for operation in ("check", "apply"):
        with pytest.raises(sources.SourceError, match="published source files differ"):
            sources.run(root, operation, "arx5-sdk")
        assert git(project, "rev-parse", "HEAD") == published
        assert git(project, "status", "--porcelain") == before


def test_published_snapshot_still_checks_original_patch_identity(snapshot):
    root, _, patch, _, _ = snapshot
    publish(snapshot)
    patch.write_bytes(PATCH + b"\n")
    with pytest.raises(sources.SourceError, match="source patch SHA-256 differs"):
        sources.run(root, "check", "arx5-sdk")
