"""Verify that polled training lists stay small and retain operational context."""

from vla_control.models import Attempt, Run

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

    response = platform.client.get("/api/v1/runs?view=summary", headers=ADMIN)
    assert response.status_code == 200, response.text
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

    full = platform.client.get("/api/v1/runs", headers=ADMIN)
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
