"""Real HTTP + independent Agent/Runner process acceptance, with a CPU fixture."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vla_bootstrap_fixture", ROOT / "scripts" / "bootstrap_demo.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def eventually(fn, check=lambda value: bool(value), timeout=25):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        try:
            last = fn()
            if check(last):
                return last
        except httpx.TransportError:
            pass
        time.sleep(0.15)
    raise AssertionError(f"Timed out; last observation: {last}")


@pytest.fixture(scope="module")
def platform(tmp_path_factory):
    base = tmp_path_factory.mktemp("platform-http")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    fixture = bootstrap.prepare_demo(base, port, create_user=True)
    # Discovery must inspect this registry without importing or executing it.
    recipe_code = base / "recipe-code"
    recipe_source = recipe_code / "src/openpi/training/config.py"
    recipe_source.parent.mkdir(parents=True)
    recipe_source.write_text(
        "raise RuntimeError('Recipe suggestions must never execute config.py')\n"
        "_CONFIGS = [TrainConfig(name='pi0_alpha', batch_size=72, num_train_steps=60000, "
        "lr_schedule=CosineDecaySchedule(peak_lr=0.00003), "
        "data=MultiDataConfigFactory(state_pad_dim=128, weights=[1, 1], "
        "datasets=[UmiData(repo_id='/fixture/first', image_horizon=1), UmiData(repo_id='/fixture/second', image_horizon=1)])), "
        "TrainConfig(name='pi0_beta')]\n"
    )
    for arguments in (["init"], ["add", "."], ["-c", "user.name=Recipe fixture", "-c", "user.email=fixture@localhost",
                      "-c", "commit.gpgsign=false", "commit", "-m", "Record static recipe fixture"]):
        subprocess.run(["git", "-C", str(recipe_code), *arguments], check=True, capture_output=True)
    fixture["config"]["profiles"].append({
        "id": "recipe-discovery", "name": "Static recipe discovery fixture", "code_dir": str(recipe_code),
        "python": sys.executable, "adapter": "openpi-sft", "execution_mode": "native",
        "metadata": {"validated_recipes": ["pi0_alpha"]},
    })
    processor_code = base / "processor-code"
    processor_source = processor_code / "src/openpi/training/config.py"
    processor_source.parent.mkdir(parents=True)
    for parent in (processor_source.parent, processor_source.parent.parent):
        (parent / "__init__.py").write_text("")
    processor_source.write_text('''from __future__ import annotations
import abc
import dataclasses

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None

@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict | None = None
    prompt_from_task: bool = False
    action_sequence_keys: tuple[str, ...] = ("actions",)

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = ""
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: DataConfig | None = None
    @abc.abstractmethod
    def create(self, assets_dir, model):
        pass

@dataclasses.dataclass(frozen=True)
class RecipeProcessor(DataConfigFactory):
    """Processor used by the existing recipe."""
    image_horizon: int = 1
    def create(self, assets_dir, model):
        return dataclasses.replace(self.base_config or DataConfig(), repo_id=self.repo_id)

@dataclasses.dataclass(frozen=True)
class AlternateProcessor(DataConfigFactory):
    """Another processor, never referenced by the selected recipe."""
    image_horizon: int = 2
    use_delta_actions: bool = True
    action_sequence_keys: tuple[str, ...] = ("actions",)
    def create(self, assets_dir, model):
        return dataclasses.replace(self.base_config or DataConfig(), repo_id=self.repo_id)

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str
    data: DataConfigFactory

_CONFIGS = [TrainConfig(name="original", data=RecipeProcessor())]
''')
    for arguments in (["init"], ["add", "."], ["-c", "user.name=Processor fixture", "-c", "user.email=fixture@localhost",
                      "-c", "commit.gpgsign=false", "commit", "-m", "Record independent processor fixture"]):
        subprocess.run(["git", "-C", str(processor_code), *arguments], check=True, capture_output=True)
    fixture["config"]["profiles"].append({
        "id": "processor-catalog", "name": "Processor catalog fixture", "code_dir": str(processor_code),
        "python": sys.executable, "adapter": "openpi-sft", "execution_mode": "native", "metadata": {},
    })
    fixture["config_file"].write_text(json.dumps(fixture["config"]))
    env = fixture["env"]
    env["VLA_SCHEDULER_INTERVAL_SECONDS"] = "0.1"
    logs = []
    children = []
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5)
    try:
        for name, argv in [("api", [sys.executable, "-m", "vla_control.main"]),
                           ("agent", [sys.executable, "-m", "vla_agent.main", "--config", str(fixture["config_file"])])]:
            log = (base / f"{name}.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen(argv, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            if name == "api":
                eventually(lambda: client.get("/healthz").status_code == 200)
                authentication = client.post("/api/v1/auth/login", json={
                    "username": fixture["credentials"]["admin_username"],
                    "password": fixture["credentials"]["admin_password"]})
                assert authentication.status_code == 200, authentication.text
                client.headers["X-CSRF-Token"] = authentication.json()["csrf_token"]
        node = eventually(lambda: client.get("/api/v1/nodes").json(), lambda nodes: len(nodes) == 1)[0]
        inputs = {}
        for role, path in fixture["assets"].items():
            response = client.post("/api/v1/assets", json={"name": role, "kind": role.upper(),
                                   "node_id": node["id"], "path": str(path), "version": "test-v1",
                                   "metadata": {"simulation": True}})
            assert response.is_success, response.text
            inputs[role] = response.json()["id"]
        yield client, fixture, node, inputs
    finally:
        client.close()
        for child in reversed(children):
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


def preview(platform, steps=5):
    client, fixture, node, inputs = platform
    response = client.post("/api/v1/run-previews", json={"name": "CPU 集成验收", "node_id": node["id"],
        "profile_id": "local-smoke", "template_id": "smoke", "recipe": "smoke", "inputs": inputs,
        "parameters": {"steps": steps, "batch_size": 4, "learning_rate": 0.01}, "gpu_count": 0})
    assert response.is_success, response.text
    result = eventually(lambda: client.get("/api/v1/run-previews/" + response.json()["id"]).json(),
                        lambda p: p["status"] != "PENDING")
    assert result["status"] == "READY", result
    assert result["request"]["recipe"] == result["code_snapshot"]["recipe"] == "smoke"
    return result


def submit(client, preflight, key):
    result = client.post("/api/v1/runs", json={"preflight_id": preflight["id"]},
                         headers={"Idempotency-Key": key})
    assert result.is_success, result.text
    return result.json()


def reauthenticate(client, fixture):
    """Sensitive writes need separate password proof; ordinary polling does not."""
    response = client.post("/api/v1/auth/reauth", json={
        "password": fixture["credentials"]["admin_password"]})
    assert response.status_code == 200, response.text


def test_real_process_training_lifecycle_and_lineage(platform):
    client, fixture, node, inputs = platform
    preflight = preview(platform)
    assert len(preflight["code_snapshot"]["commit_sha"]) == 40
    run = submit(client, preflight, "integration-success")
    assert submit(client, preflight, "integration-success")["id"] == run["id"]
    result = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                        lambda r: r["state"] in {"SUCCEEDED", "FAILED"})
    assert result["state"] == "SUCCEEDED", result
    assert result["recipe"] == result["code_expectation"]["recipe"] == "smoke"
    assert result["attempts"][0]["code_snapshot"]["recipe"] == "smoke"
    assert result["attempts"][0]["code_snapshot"]["commit_sha"] == preflight["code_snapshot"]["commit_sha"]
    assert result["outputs"] and result["outputs"][0]["metadata"]["simulation"]
    assert result["outputs"][0]["node_id"] == node["id"]
    events = client.get(f"/api/v1/runs/{run['id']}/events").json()
    assert any(event["type"] == "METRIC" for event in events)
    logs = client.get(f"/api/v1/runs/{run['id']}/logs").json()
    assert logs["items"]


def test_real_training_finishes_after_human_session_idle_expiry(platform):
    from sqlalchemy import func, select
    from vla_control.auth_models import UserSession
    from vla_control.common import offset_time
    from vla_control.db import Database
    from vla_control.models import Event, Run

    client, fixture, *_ = platform
    run = submit(client, preview(platform, steps=40), "integration-session-expiry")
    running = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                         lambda value: value["state"] == "RUNNING")
    attempt_id = running["attempts"][0]["id"]
    db = Database(fixture["env"]["VLA_DATABASE_URL"])
    try:
        with db.write() as transaction:
            previous_seq = transaction.scalar(select(func.max(Event.seq)).where(Event.attempt_id == attempt_id)) or 0
            for row in transaction.scalars(select(UserSession)):
                row.last_activity_at = offset_time(-901)
        assert client.get("/api/v1/runs/" + run["id"]).status_code == 401

        def persisted_state():
            with db.read() as transaction:
                return transaction.get(Run, run["id"]).state

        assert eventually(persisted_state, lambda state: state in {"SUCCEEDED", "FAILED"}) == "SUCCEEDED"
        assert client.get("/api/v1/runs/" + run["id"]).status_code == 401
        with db.read() as transaction:
            later = transaction.scalars(select(Event).where(Event.attempt_id == attempt_id,
                                                            Event.seq > previous_seq)).all()
            assert any(event.type == "METRIC" for event in later)
            assert any(event.type == "SUCCEEDED" for event in later)
    finally:
        db.close()
        authentication = client.post("/api/v1/auth/login", json={
            "username": fixture["credentials"]["admin_username"],
            "password": fixture["credentials"]["admin_password"]})
        assert authentication.status_code == 200, authentication.text
        client.headers["X-CSRF-Token"] = authentication.json()["csrf_token"]
    assert client.get("/api/v1/runs/" + run["id"]).json()["state"] == "SUCCEEDED"


def test_cancel_waits_for_runner_exit(platform):
    client, *_ = platform
    run = submit(client, preview(platform, steps=500), "integration-cancel")
    eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(), lambda r: r["state"] == "RUNNING")
    result = client.post(f"/api/v1/runs/{run['id']}/cancel")
    assert result.is_success, result.text
    final = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                       lambda r: r["state"] in {"CANCELLED", "FAILED"})
    assert final["state"] == "CANCELLED", final


def test_changed_code_is_blocked_before_process_start(platform):
    client, fixture, *_ = platform
    preflight = preview(platform)
    code = Path(fixture["config"]["profiles"][0]["code_dir"]) / "smoke_training.py"
    previous = code.read_text()
    try:
        code.write_text(previous + "\n# changed after preview\n")
        run = submit(client, preflight, "integration-code-drift")
        result = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                            lambda r: r["state"] == "FAILED")
        events = client.get(f"/api/v1/runs/{run['id']}/events").json()
        assert not any(event["type"] == "STARTED" for event in events)
        assert "CODE_CHANGED_SINCE_PREFLIGHT" in str(result) + str(events)
    finally:
        code.write_text(previous)


def test_changed_runtime_is_blocked_even_with_same_git_commit(platform):
    client, fixture, *_ = platform
    preflight = preview(platform)
    config_file = fixture["config_file"]
    previous = config_file.read_text()
    changed = json.loads(previous)
    changed["profiles"][0]["recipe"] = "changed-after-preflight"
    try:
        config_file.write_text(json.dumps(changed))
        run = submit(client, preflight, "integration-runtime-drift")
        result = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                            lambda r: r["state"] == "FAILED")
        events = client.get(f"/api/v1/runs/{run['id']}/events").json()
        assert not any(event["type"] == "STARTED" for event in events)
        assert "CODE_CHANGED_SINCE_PREFLIGHT" in str(result) + str(events)
    finally:
        config_file.write_text(previous)


def test_real_agent_path_completion_lists_only_machine_local_names(platform, tmp_path):
    client, fixture, node, _ = platform
    base = fixture["base"]

    def suggest(prefix):
        response = client.post(f"/api/v1/nodes/{node['id']}/path-suggestions", json={"prefix": prefix, "limit": 30})
        assert response.status_code == 202, response.text
        result = eventually(lambda: client.get(
            f"/api/v1/nodes/{node['id']}/path-suggestions/{response.json()['id']}").json(),
            lambda value: value["status"] != "PENDING")
        return result

    roots = suggest("")
    assert roots["status"] == "READY", roots
    assert any(entry["path"] == str(base) + "/" for entry in roots["entries"])
    names = suggest(str(base) + "/demo-a")
    assert names["status"] == "READY", names
    assert {entry["name"] for entry in names["entries"]} == {"demo-assets"}
    assert names["entries"][0]["kind"] == "directory"
    nested = suggest(str(base / "demo-assets") + "/")
    assert nested["status"] == "READY", nested
    assert {entry["name"] for entry in nested["entries"]} == {"dataset", "model", "norm"}
    named_file = base / "autocomplete-marker.txt"
    named_file.write_text("NEVER_RETURN_FILE_CONTENT")
    escape = base / "autocomplete-escape"
    escape.symlink_to(tmp_path, target_is_directory=True)
    matches = suggest(str(base) + "/autocomplete-")
    assert matches["status"] == "READY", matches
    assert {entry["name"] for entry in matches["entries"]} == {"autocomplete-marker.txt"}
    assert "NEVER_RETURN_FILE_CONTENT" not in json.dumps(matches)
    rejected = suggest(str(escape) + "/")
    assert rejected["status"] == "FAILED", rejected
    assert rejected["error"]["code"] == "PATH_OUTSIDE_ROOTS"


def test_real_agent_entrypoint_completion_is_project_scoped(platform):
    client, fixture, node, _ = platform
    project = fixture["base"] / "recipe-code"
    scripts = project / "scripts"
    scripts.mkdir()
    outside = fixture["base"] / "outside-entrypoint.py"
    outside.write_text("raise AssertionError('must not execute outside file')")
    (scripts / "train_scope.py").write_text("raise AssertionError('suggestions never execute files')")
    (scripts / "train_scope.sh").write_text("not a Python entrypoint")
    (scripts / "train_escape.py").symlink_to(outside)
    escape = project / "escape-directory"
    escape.symlink_to(fixture["base"], target_is_directory=True)
    try:
        def suggest(prefix):
            response = client.post(f"/api/v1/nodes/{node['id']}/path-suggestions", json={
                "profile_id": "recipe-discovery", "prefix": prefix, "limit": 30})
            assert response.status_code == 202, response.text
            return eventually(lambda: client.get(
                f"/api/v1/nodes/{node['id']}/path-suggestions/{response.json()['id']}").json(),
                lambda result: result["status"] != "PENDING")

        root = suggest("")
        assert root["status"] == "READY" and root["base_path"] == str(project)
        assert root["profile_id"] == "recipe-discovery"
        assert all(entry["path"].startswith(str(project) + "/") for entry in root["entries"])
        assert "escape-directory" not in {entry["name"] for entry in root["entries"]}
        files = suggest(str(scripts) + "/train_")
        assert files["status"] == "READY", files
        assert {entry["name"] for entry in files["entries"]} == {"train_scope.py"}
        assert "raise AssertionError" not in json.dumps(files)
        escaped = suggest(str(escape) + "/")
        assert escaped["status"] == "FAILED" and escaped["error"]["code"] == "PATH_OUTSIDE_PROJECT"
        rejected = client.post(f"/api/v1/nodes/{node['id']}/path-suggestions", json={
            "profile_id": "recipe-discovery", "prefix": str(fixture["base"]) + "/", "limit": 30})
        assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "PATH_OUTSIDE_PROJECT"
    finally:
        for item in scripts.iterdir():
            item.unlink()
        scripts.rmdir()
        escape.unlink()
        outside.unlink()


def test_real_agent_recipe_completion_preserves_validation_boundary(platform):
    client, _, node, _ = platform
    response = client.post(f"/api/v1/nodes/{node['id']}/recipe-suggestions", json={
        "profile_id": "recipe-discovery", "prefix": "pi0_", "limit": 30,
    })
    assert response.status_code == 202, response.text
    result = eventually(lambda: client.get(
        f"/api/v1/nodes/{node['id']}/recipe-suggestions/{response.json()['id']}").json(),
        lambda value: value["status"] != "PENDING")
    assert result["status"] == "READY", result
    entries = {entry["name"]: entry for entry in result["entries"]}
    assert set(entries) == {"pi0_alpha", "pi0_beta"}
    assert entries["pi0_alpha"]["validated"] is True
    assert entries["pi0_beta"]["validated"] is False
    assert entries["pi0_alpha"]["source"] == "both"
    assert entries["pi0_beta"]["source"] == "code"
    assert result["source"] == "src/openpi/training/config.py"
    # A keyword in the middle of a name is a search, not just a prefix completion.
    for keyword in ("alpha", "ALPHA"):
        response = client.post(f"/api/v1/nodes/{node['id']}/recipe-suggestions", json={
            "profile_id": "recipe-discovery", "prefix": keyword, "limit": 30,
        })
        assert response.status_code == 202, response.text
        matched = eventually(lambda: client.get(
            f"/api/v1/nodes/{node['id']}/recipe-suggestions/{response.json()['id']}").json(),
            lambda value: value["status"] != "PENDING")
        assert matched["status"] == "READY", matched
        assert [entry["name"] for entry in matched["entries"]] == ["pi0_alpha"]
        assert matched["entries"][0]["validated"] is True


def test_real_agent_lists_processors_outside_the_selected_recipe(platform):
    client, fixture, node, _ = platform
    source = fixture["base"] / "processor-code/src/openpi/training/config.py"
    original = source.read_bytes()
    response = client.post(f"/api/v1/nodes/{node['id']}/data-processors", json={"profile_id": "processor-catalog"})
    assert response.status_code == 202, response.text
    result = eventually(lambda: client.get(
        f"/api/v1/nodes/{node['id']}/data-processors/{response.json()['id']}").json(),
        lambda value: value["status"] != "PENDING", timeout=40)
    assert result["status"] == "READY", result
    assert result["profile_id"] == "processor-catalog"
    entries = {entry["name"]: entry for entry in result["catalog"]["entries"]}
    assert {"RecipeProcessor", "AlternateProcessor"} <= set(entries)
    alternate = entries["AlternateProcessor"]
    assert alternate["available"] is True
    parameters = {item["path"]: item for item in alternate["parameters"]}
    assert parameters["image_horizon"]["default"] == 2
    assert parameters["use_delta_actions"]["default"] is True
    assert parameters["base_config.prompt_from_task"]["default"] is False
    assert not any("repo_id" in key or "assets" in key for key in parameters)
    assert source.read_bytes() == original


def test_real_agent_recipe_detail_clone_and_revisions_do_not_execute_source(platform):
    client, fixture, node, _ = platform
    response = client.post(f"/api/v1/nodes/{node['id']}/recipe-details", json={
        "profile_id": "recipe-discovery", "recipe": "pi0_alpha",
    })
    assert response.status_code == 202, response.text
    query = eventually(lambda: client.get(
        f"/api/v1/nodes/{node['id']}/recipe-details/{response.json()['id']}").json(),
        lambda value: value["status"] != "PENDING")
    assert query["status"] == "READY", query
    values = {field["path"]: field.get("value") for field in query["detail"]["fields"] if field["kind"] == "literal"}
    assert values["batch_size"] == 72 and values["lr_schedule.peak_lr"] == 0.00003
    code_before = (fixture["base"] / "recipe-code/src/openpi/training/config.py").read_bytes()
    saved = client.post("/api/v1/recipes", json={"node_id": node["id"], "profile_id": "recipe-discovery",
        "name": "集成测试：低学习率", "description": "Static fixture, never a VLA training result",
        "detail_query_id": query["id"], "overrides": {"batch_size": 36, "lr_schedule.peak_lr": 0.00002}})
    assert saved.status_code == 201, saved.text
    first = saved.json()
    assert first["latest_revision"]["revision"] == 1 and first["validated"] is False
    rejected = client.post("/api/v1/run-previews", json={"name": "Automatic preflight requires real resources", "node_id": node["id"],
        "profile_id": "recipe-discovery", "template_id": "openpi-sft", "recipe": "pi0_alpha",
        "recipe_revision_id": first["latest_revision"]["id"], "gpu_count": 1})
    # Saved revisions no longer need a manual acceptance flag. This CPU-only
    # fixture still cannot request a GPU it does not have.
    assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "GPU_CAPACITY_EXCEEDED"
    revised = client.post(f"/api/v1/recipes/{first['id']}/revisions", json={
        "detail_query_id": query["id"], "overrides": {"batch_size": 48}})
    assert revised.status_code == 201, revised.text
    assert revised.json()["latest_revision"]["revision"] == 2
    historical = client.get("/api/v1/recipe-revisions/" + first["latest_revision"]["id"]).json()
    assert historical["definition"] == first["latest_revision"]["definition"]
    assert (fixture["base"] / "recipe-code/src/openpi/training/config.py").read_bytes() == code_before


def test_real_agent_writes_nested_recipe_and_commits_once_without_executing_source(platform):
    import ast

    client, fixture, node, _ = platform
    checkout = fixture["base"] / "recipe-code"
    source = checkout / "src/openpi/training/config.py"
    before = source.read_text()
    old_commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    response = client.post(f"/api/v1/nodes/{node['id']}/recipe-details", json={
        "profile_id": "recipe-discovery", "recipe": "pi0_alpha"})
    assert response.status_code == 202, response.text
    query = eventually(lambda: client.get(
        f"/api/v1/nodes/{node['id']}/recipe-details/{response.json()['id']}").json(),
        lambda value: value["status"] != "PENDING")
    assert query["status"] == "READY", query
    saved = client.post("/api/v1/recipes", json={
        "node_id": node["id"], "profile_id": "recipe-discovery", "name": "写入代码的嵌套配方",
        "detail_query_id": query["id"], "overrides": {
            "batch_size": 40, "data.state_pad_dim": 64, "data.datasets[0].image_horizon": 2}})
    assert saved.status_code == 201, saved.text
    original_revision = saved.json()["latest_revision"]
    endpoint = f"/api/v1/recipe-revisions/{original_revision['id']}/writeback"
    denied = client.post(endpoint, json={"config_name": "pi0_saved_committed"})
    assert denied.status_code == 403 and denied.json()["error"]["code"] == "REAUTH_REQUIRED"
    assert source.read_text() == before
    reauthenticate(client, fixture)
    response = client.post(endpoint, json={"config_name": "pi0_saved_committed"})
    assert response.is_success, response.text
    job_id = response.json()["id"]
    result = eventually(lambda: client.get(f"/api/v1/recipe-writebacks/{job_id}").json(),
                        lambda value: value["status"] != "PENDING")
    assert result["status"] == "SUCCEEDED", result
    revision = result["recipe"]["latest_revision"]
    assert revision["id"] != original_revision["id"]
    assert revision["definition"]["base_recipe"] == "pi0_saved_committed"
    assert revision["definition"]["overrides"] == {}
    assert revision["definition"]["source"]["commit_sha"] == result["commit_sha"] != old_commit
    assert client.get(f"/api/v1/recipe-revisions/{original_revision['id']}").json()["definition"] == original_revision["definition"]
    content = source.read_text()
    assert "raise RuntimeError('Recipe suggestions must never execute config.py')" in content
    assert before.split("_CONFIGS =", 1)[0] == content.split("_CONFIGS =", 1)[0]
    tree = ast.parse(content)
    calls = {next(keyword.value.value for keyword in call.keywords if keyword.arg == "name"): call
             for call in ast.walk(tree) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
             and call.func.id == "TrainConfig"}
    new = calls["pi0_saved_committed"]
    values = {keyword.arg: keyword.value for keyword in new.keywords}
    assert ast.literal_eval(values["batch_size"]) == 40
    data = {keyword.arg: keyword.value for keyword in values["data"].keywords}
    assert ast.literal_eval(data["state_pad_dim"]) == 64
    assert ast.literal_eval(data["weights"]) == [1, 1]
    children = [{keyword.arg: ast.literal_eval(keyword.value) for keyword in child.keywords}
                for child in data["datasets"].elts]
    assert [child["repo_id"] for child in children] == ["/fixture/first", "/fixture/second"]
    assert [child["image_horizon"] for child in children] == [2, 1]
    changed = subprocess.check_output(["git", "-C", str(checkout), "diff-tree", "--no-commit-id", "--name-only", "-r", result["commit_sha"]], text=True).splitlines()
    assert changed == ["src/openpi/training/config.py"]
    retry = client.post(endpoint, json={"config_name": "pi0_saved_committed"})
    assert retry.is_success and retry.json()["id"] == job_id, retry.text
    assert subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() == result["commit_sha"]


@pytest.mark.parametrize("interpreter_valid", [True, False], ids=["configured-python", "missing-python"])
def test_generated_onboarding_config_is_checked_by_the_real_agent(platform, tmp_path, interpreter_valid):
    client, fixture, *_ = platform
    reauthenticate(client, fixture)
    original = fixture["config"]
    python = sys.executable if interpreter_valid else str(tmp_path / "missing-environment/bin/python")
    response = client.post("/api/v1/agent-configs", json={
        "name": "接入向导环境验收 " + tmp_path.name,
        "roots": [*original["roots"], str(tmp_path)],
        "output_root": str(tmp_path / "outputs"),
        "state_dir": str(tmp_path / "state"),
        "profiles": [{
            "id": "web-configured", "name": "网页登记的 Python 环境",
            "python": python, "code_dir": original["profiles"][0]["code_dir"],
            "adapter": "smoke", "execution_mode": "native",
        }],
    })
    assert response.status_code == 201, response.text
    generated = response.json()
    assert generated["config"]["profiles"][0]["recipe"] is None
    assert generated["config"]["control_plane_url"] == original["control_plane_url"]
    config = tmp_path / generated["filename"]
    config.write_text(json.dumps(generated["config"]))
    config.chmod(0o600)
    with (tmp_path / "onboarding-agent.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-m", "vla_agent.main", "--config", str(config)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            def locate_node():
                profiles = client.get("/api/v1/profiles").json()
                profile = next((p for p in profiles if p["metadata"].get("onboarding_id") == generated["id"]), None)
                if not profile:
                    return None
                return client.get("/api/v1/nodes/" + profile["node_id"]).json()
            node = eventually(locate_node, lambda n: n and n["status"] == "ONLINE"
                              and n.get("metrics", {}).get("runtime_checks")
                              and n["metrics"]["runtime_checks"][0]["status"] != "CHECKING", timeout=30)
            check = node["metrics"]["runtime_checks"][0]
            assert node["profiles"][0]["recipe"] is None
            assert check["profile_id"] == "web-configured"
            assert check["status"] == ("READY" if interpreter_valid else "FAILED"), check
            if interpreter_valid:
                assert check["executable"] == python
                assert check["python_version"]
            else:
                assert check["error"] and "Python" in check["error"]
            preview = client.post("/api/v1/run-previews", json={
                "name": "按网页配置解释器预检", "node_id": node["id"], "profile_id": "web-configured",
                "template_id": "smoke", "inputs": {}, "parameters": {"steps": 2}, "gpu_count": 0,
            })
            assert preview.is_success, preview.text
            result = eventually(lambda: client.get("/api/v1/run-previews/" + preview.json()["id"]).json(),
                                lambda p: p["status"] != "PENDING")
            assert result["status"] == ("READY" if interpreter_valid else "FAILED"), result
            if interpreter_valid:
                assert result["request"]["recipe"] == result["code_snapshot"]["recipe"] == "smoke"
                run = submit(client, result, "generated-no-default-recipe-" + generated["id"])
                finished = eventually(lambda: client.get("/api/v1/runs/" + run["id"]).json(),
                                      lambda r: r["state"] in {"SUCCEEDED", "FAILED"})
                assert finished["state"] == "SUCCEEDED", finished
                assert finished["recipe"] == finished["attempts"][0]["code_snapshot"]["recipe"] == "smoke"
            else:
                assert result["error"]["code"] == "RUNTIME_UNAVAILABLE"
        finally:
            child.terminate()
            child.wait(timeout=10)
