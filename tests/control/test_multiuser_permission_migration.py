"""The role rollout adds metadata without deleting sessions or machine history."""

from sqlalchemy import inspect, select

from vla_control.common import now
from vla_control.deployment_record_models import DeploymentRecord
from vla_control.models import Command, GPUReservation, Preview, Run

from .conftest import ADMIN
from .test_dataset_norm_migration import _legacy_table, _snapshot


def test_permission_upgrade_preserves_rows_sessions_and_uncertain_deliveries(platform):
    node = platform.register()
    run = platform.run(node)
    assert platform.schedule() == 1
    password = "ownership-migration-password"
    user = platform.app.state.auth.create_user("migration-admin", password, "admin")
    with platform.app.state.db.write() as session:
        session.add(GPUReservation(node_id=node["node_id"], gpu_uuid="GPU-1", user_id=user["id"], created_at=now()))
    platform.client.base_url = "https://testserver"
    login = platform.client.post("/api/v1/auth/login", json={"username": "migration-admin", "password": password})
    assert login.status_code == 200
    database = platform.app.state.db
    removals = {"asset_transfers": {"owner_user_id", "execution_user_id"},
        "checkpoint_transfers": {"owner_user_id", "execution_user_id"},
        "policy_deployments": {"execution_user_id"},
        "deployment_records": {"owner_user_id"}, "previews": {"owner_user_id"},
        "gpu_reservations": {"expires_at"},
        "commands": {"delivered_at", "actor_user_id", "required_permission"},
        "remote_deployments": {"actor_user_id"}}
    projections = {}
    raw = database.engine.raw_connection()
    try:
        raw.commit()
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        for name, columns in removals.items():
            retained = _legacy_table(cursor, name, columns)
            projections[name] = 'SELECT ' + ','.join('"' + value + '"' for value in retained) + f' FROM {name}'
        raw.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        raw.close()
    projections.update({name: f"SELECT * FROM {name}" for name in ("users", "user_sessions", "nodes", "runs", "attempts")})
    before = _snapshot(database, projections)
    database.initialize()
    database.initialize()
    assert _snapshot(database, projections) == before
    for name, columns in removals.items():
        assert columns <= {column["name"] for column in inspect(database.engine).get_columns(name)}
    with database.read() as session:
        assert session.get(Run, run["id"]).owner_user_id is None
        assert session.scalar(select(Preview)).owner_user_id is None
        for row in session.scalars(select(Command)):
            assert row.delivered_at == row.created_at
            assert row.actor_user_id is None and row.required_permission is None
        assert session.get(GPUReservation, (node["node_id"], "GPU-1")).expires_at is None
    assert platform.client.get("/api/v1/auth/me").status_code == 200
