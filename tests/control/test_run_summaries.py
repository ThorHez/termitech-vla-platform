"""Verify that polled training lists stay small and retain operational context."""

from vla_control.models import Asset, Attempt, Event, Profile, Run

from .conftest import ADMIN


EARLY = "2026-11-10T08:00:00+00:00"
LATE = "2026-11-11T08:00:00+00:00"


def add_run(session, node, identifier, **changes):
    values = {
        "id": identifier,
        "name": f"训练 {identifier}",
        "node_id": node["node_id"],
        "profile_id": "default",
        "template_id": "openpi-sft",
        "state": "FAILED",
        "created_at": EARLY,
        "updated_at": LATE,
        "finished_at": LATE,
        "inputs": {},
        "parameters": {"steps": 100, "batch_size": 32},
        "gpu_count": 2,
        "code_expectation": {},
    }
    session.add(Run(**{**values, **changes}))


def test_summary_lists_project_existing_state_progress_and_lineage(platform):
    node = platform.register()
    other = platform.register(name="other")
    large_config = "generated-config-details " * 4096
    with platform.app.state.db.write() as session:
        add_run(session, node, "main", created_at=LATE,
                inputs={"dataset": "dataset-a", "dataset_0": "dataset-a",
                        "dataset_1": "dataset-b", "model": "model"},
                code_expectation={"automatic_preflight": {"plan": {"generated_config": large_config}},
                                  "dataset_bindings": [{"dataset_asset_id": "dataset-a"},
                                                       {"dataset_asset_id": "dataset-b"}]})
        add_run(session, node, "legacy", inputs={"dataset": "registered-dataset"}, code_expectation={
            "dataset_bindings": [{"dataset_asset_id": "legacy-dataset"}], "requested_entrypoint": "",
        })
        add_run(session, other, "elsewhere", code_expectation={"requested_entrypoint": "scripts/custom.py"},
                parameters={"steps": "not a number"})
        add_run(session, node, "hidden", deleted_at=LATE)
        session.flush()
        session.add_all([
            Attempt(id="formerly", run_id="main", node_id=node["node_id"], execution_token="token-1",
                    number=1, state="FAILED", code_changed_during_run=True, created_at=EARLY,
                    latest_metrics={"step": 55, "total_steps": 100}),
            Attempt(id="latest", run_id="main", node_id=node["node_id"], execution_token="token-2",
                    number=2, state="FAILED", code_changed_during_run=False, created_at=LATE,
                    latest_metrics={"step": 0, "total_steps": 101, "detail": large_config}),
        ])

    response = platform.client.get("/api/v1/runs", headers=ADMIN)
    assert response.status_code == 200, response.text
    explicit_summary = platform.client.get("/api/v1/runs?view=summary", headers=ADMIN)
    assert explicit_summary.status_code == 200
    assert explicit_summary.content == response.content
    rows = response.json()
    assert [row["id"] for row in rows] == ["main", "legacy", "elsewhere"]
    assert rows[0] == {
        "id": "main", "name": "训练 main", "node_id": node["node_id"], "profile_id": "default",
        "template_id": "openpi-sft", "state": "FAILED", "created_at": LATE, "updated_at": LATE,
        "finished_at": LATE, "owner_user_id": None, "gpu_count": 2,
        "dataset_asset_ids": ["dataset-a", "dataset-b"], "parameters": {"steps": 100},
        "resume_candidate": False,
        "attempts": [
            {"id": "formerly", "state": "FAILED", "code_changed_during_run": True},
            {"id": "latest", "state": "FAILED", "code_changed_during_run": False,
             "latest_metrics": {"step": 0, "total_steps": 101}},
        ],
    }
    assert rows[1]["dataset_asset_ids"] == ["registered-dataset", "legacy-dataset"]
    assert rows[1]["resume_candidate"] is True
    assert rows[2]["resume_candidate"] is False
    assert rows[2]["parameters"] == {}

    full = platform.client.get("/api/v1/runs?view=full", headers=ADMIN)
    detail = platform.client.get("/api/v1/runs/main", headers=ADMIN)
    assert detail.status_code == full.status_code == 200
    assert detail.json() == full.json()[0]
    assert detail.json()["code_expectation"]["automatic_preflight"]["plan"]["generated_config"] == large_config
    assert len(response.content) * 20 < len(full.content)
    assert "scripts/custom.py" not in response.text

    machine = platform.client.get(f"/api/v1/nodes/{node['node_id']}?view=summary", headers=ADMIN)
    assert machine.status_code == 200, machine.text
    assert machine.json()["runs"] == rows[:2]
    default_machine = platform.client.get(f"/api/v1/nodes/{node['node_id']}", headers=ADMIN)
    assert default_machine.status_code == 200
    assert default_machine.json()["runs"][0] == detail.json()

    page = platform.client.get("/api/v1/background-tasks?view=summary&page_size=1", headers=ADMIN)
    assert page.status_code == 200, page.text
    assert page.json()["items"] == [{"key": "training:main", "kind": "training", "item": rows[0]}]
    assert page.json()["total"] == 3
    default_page = platform.client.get("/api/v1/background-tasks?page_size=1", headers=ADMIN)
    assert default_page.status_code == 200
    assert default_page.json()["items"][0]["item"] == detail.json()


def test_node_summary_keeps_exact_active_counts_independent_of_history_limit(platform):
    node = platform.register()
    other = platform.register(name="other")
    with platform.app.state.db.write() as session:
        session.add(Profile(node_id=node["node_id"], id="legacy-runner", name="legacy runtime",
                            code_dir="/machine/legacy", python="/usr/bin/python3", adapter="smoke"))
        session.flush()
        for index in range(100):
            add_run(session, node, f"finished-{index:03}", created_at=LATE)
        add_run(session, node, "older-running", state="RUNNING")
        add_run(session, node, "older-starting", state="STARTING")
        add_run(session, node, "older-unknown", state="UNKNOWN", profile_id="legacy-runner")
        add_run(session, node, "deleted-active", state="QUEUED", deleted_at=LATE)
        add_run(session, other, "other-active", state="RUNNING")

    path = f"/api/v1/nodes/{node['node_id']}"
    for limit in (None, 1, 10, 500):
        params = {"view": "summary"}
        if limit is not None:
            params["run_limit"] = limit
        response = platform.client.get(path, headers=ADMIN, params=params)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["active_training"] == {"count": 3, "profile_ids": ["default", "legacy-runner"]}
        assert len(result["runs"]) == {None: 100, 1: 1, 10: 10, 500: 103}[limit]
        assert all(row["node_id"] == node["node_id"] for row in result["runs"])
        assert result["runs"][0]["id"] == "finished-099"
    assert platform.client.get(path, headers=ADMIN, params={"view": "summary", "run_limit": 501}).status_code == 422


def test_node_asset_limit_returns_recent_catalog_cards_with_compact_norm_references(platform):
    node = platform.register()["node_id"]
    other = platform.register(name="other")["node_id"]

    def asset(identifier, **changes):
        fields = {
            "id": identifier, "node_id": node, "name": identifier, "kind": "MODEL",
            "path": f"/machine/{identifier}", "version": "v1", "state": "REGISTERED",
            "created_at": LATE, "metadata_json": {"detail": "large payload " * 500},
        }
        return Asset(**(fields | changes))

    with platform.app.state.db.write() as session:
        session.add_all([
            asset("z-free-norm", kind="NORM"),
            asset("zz-hidden-bound-norm", kind="NORM"),
            asset("older-live-norm", kind="NORM", created_at=EARLY),
            asset("older-archived-norm", kind="NORM", state="ARCHIVED", created_at=EARLY),
        ])
        session.flush()
        session.add_all([
            asset("zz-other-node", node_id=other),
            asset("zz-archived-dataset", kind="DATASET", state="ARCHIVED", norm_asset_id="z-free-norm"),
            asset("y-dataset-archived-norm", kind="DATASET", norm_asset_id="older-archived-norm"),
            asset("x-dataset-live-norm", kind="DATASET", norm_asset_id="older-live-norm"),
            asset("w-dataset-live-norm", kind="DATASET", norm_asset_id="older-live-norm"),
            asset("older-dataset-beyond-limit", kind="DATASET", created_at=EARLY, norm_asset_id="zz-hidden-bound-norm"),
            *(asset(f"model-{index:02}") for index in range(1, 11)),
        ])

    path = f"/api/v1/nodes/{node}"
    limited = platform.client.get(path, params={"view": "summary", "asset_limit": 10, "run_limit": 10}, headers=ADMIN)
    assert limited.status_code == 200, limited.text
    result = limited.json()
    assert [asset["id"] for asset in result["assets"]] == [
        "z-free-norm", "y-dataset-archived-norm", "x-dataset-live-norm", "w-dataset-live-norm",
        *(f"model-{index:02}" for index in range(10, 4, -1)),
    ]
    assert {asset["id"]: asset["state"] for asset in result["asset_norms"]} == {
        "older-live-norm": "REGISTERED", "older-archived-norm": "ARCHIVED",
    }
    assert all(set(asset) == {"id", "name", "kind", "node_id", "path", "version", "state", "created_at", "norm_asset_id"}
               for asset in result["assets"] + result["asset_norms"])

    for params in ({}, {"view": "summary"}):
        unlimited = platform.client.get(path, params=params, headers=ADMIN).json()
        assert "asset_norms" not in unlimited
        by_id = {item["id"]: item for item in unlimited["assets"]}
        assert "zz-other-node" not in by_id
        assert by_id["zz-archived-dataset"]["state"] == "ARCHIVED"
        assert by_id["zz-hidden-bound-norm"]["metadata"]["detail"] == "large payload " * 500
    for invalid_limit in (0, 101):
        assert platform.client.get(path, params={"asset_limit": invalid_limit}, headers=ADMIN).status_code == 422


def test_run_status_and_timeline_only_return_fields_used_by_live_view(platform):
    node = platform.register()
    with platform.app.state.db.write() as session:
        add_run(session, node, "live", state="RUNNING", queue_reason="AWAITING_AGENT",
                code_expectation={"generated_config": "large config " * 1000})
        session.flush()
        session.add(Attempt(id="current", run_id="live", node_id=node["node_id"], execution_token="live-token",
                            state="RUNNING", created_at=EARLY,
                            latest_metrics={"step": 12, "loss": 1.2, "timestamp": LATE, "raw": "large text " * 1000},
                            resources={"rss_bytes": 500, "cpu_seconds": 2, "diagnostic": "large text " * 1000}))
        session.flush()
        for seq, kind in enumerate(("STARTED", "LOG", "METRIC", "RESOURCE", "ARTIFACT"), 1):
            session.add(Event(attempt_id="current", seq=seq, type=kind, timestamp=LATE,
                              data={"payload": "large text " * 1000}, digest=str(seq)))

    status = platform.client.get("/api/v1/runs/live?view=status", headers=ADMIN)
    assert status.status_code == 200, status.text
    assert status.json() == {
        "id": "live", "state": "RUNNING", "updated_at": LATE, "queue_reason": "AWAITING_AGENT",
        "attempt": {"id": "current", "latest_metrics": {"step": 12, "loss": 1.2, "timestamp": LATE},
                    "resources": {"rss_bytes": 500, "cpu_seconds": 2}},
    }
    assert status.json()["state"] == platform.client.get("/api/v1/runs/live", headers=ADMIN).json()["state"]

    path = "/api/v1/runs/live/events"
    default = platform.client.get(path, headers=ADMIN)
    assert [item["seq"] for item in default.json()] == [1, 2, 3, 4, 5]
    timeline = platform.client.get(path + "?view=timeline&after_seq=1", headers=ADMIN)
    assert timeline.status_code == 200, timeline.text
    assert timeline.json() == [{"attempt_id": "current", "seq": 5, "type": "ARTIFACT", "timestamp": LATE, "data": {}}]
