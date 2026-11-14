"""Registration removal and permanent file deletion use different authority."""

from sqlalchemy import select

from vla_agent.asset_purge import execute_asset_purge_command
from vla_agent.state import State
from vla_control.asset_deletions import INSPECT, PURGE
from vla_control.common import offset_time
from vla_control.models import Asset, Command, Node, Profile
from .test_dataset_archive_permissions import archive_path, registered
from .test_dataset_registration_permissions import body
from .test_multiuser_task_permissions import PASSWORD, team  # noqa: F401

LEDGER = "/api/v1/assets/deletions"
PIN = "a" * 64


def path(asset, action):
    return f"/api/v1/assets/{asset['id']}/{action}"


def enable(platform, node):
    with platform.app.state.db.write() as session:
        session.get(Node, node["node_id"]).metrics = {"capabilities": [INSPECT, PURGE]}


def reauth(admin):
    response = admin.post("/api/v1/auth/reauth", json={"password": PASSWORD})
    assert response.status_code == 200, response.text
    admin.headers["X-CSRF-Token"] = response.json()["csrf_token"]


def ack(platform, node, command_id, result, status="APPLIED"):
    return platform.client.post(f"/agent/v1/commands/{command_id}/ack", headers=node["headers"],
                                json={"node_id": node["node_id"], "status": status, "result": result})


def inspect(team, asset, status="READY"):
    platform, node, _, clients = team
    admin = clients["team-admin"]
    response = admin.post(path(asset, "purge-inspections"))
    assert response.status_code == 202, response.text
    receipt = response.json()
    assert admin.post(path(asset, "purge-inspections")).json()["id"] == receipt["id"]
    commands = platform.commands(node)
    command, = commands
    assert command["id"] == receipt["id"] and command["type"] == INSPECT
    assert command["payload"]["asset_id"] == asset["id"] and command["payload"]["path"] == asset["path"]
    assert isinstance(command["payload"]["protected_paths"], list)
    result = {"asset_id": asset["id"], "path": asset["path"], "status": status,
              "identity": PIN if status == "READY" else None}
    assert ack(platform, node, receipt["id"], result).status_code == 200
    return {"confirm_path": asset["path"], "inspection_id": receipt["id"]}


def test_archive_only_records_owner_and_never_queues_file_work(team):
    platform, node, _, clients = team
    admin, operator = clients["team-admin"], clients["alice"]
    asset = registered(team, kind="NORM")
    assert clients["bob"].post(archive_path(asset)).status_code == 403
    assert admin.get(LEDGER).json() == []
    assert operator.post(archive_path(asset)).status_code == 200
    item, = admin.get(LEDGER).json()
    assert item["id"] == asset["id"] and item["archived_by"] == "alice" and item["archived_at"]
    assert item["file_deletion"] is None and item["inspection"] is None and item["can_purge"] is False
    assert "Agent" in item["purge_block_reason"]
    assert admin.post(archive_path(asset)).status_code == 200
    assert admin.get(LEDGER).json() == [item]
    assert platform.commands(node) == []
    assert operator.get(LEDGER).status_code == 403
    assert operator.post(path(asset, "purge-inspections")).status_code == 403
    assert operator.post(path(asset, "purge"), json={"confirm_path": asset["path"], "inspection_id": asset["id"]}).status_code == 403


def test_purge_requires_password_confirmation_and_matching_agent_result(team):
    platform, node, _, clients = team
    enable(platform, node)
    admin = clients["team-admin"]
    asset = registered(team, kind="NORM")
    assert clients["alice"].post(archive_path(asset)).status_code == 200
    body = inspect(team, asset)
    result = admin.post(path(asset, "purge"), json=body)
    assert result.status_code == 403 and result.json()["error"]["code"] == "REAUTH_REQUIRED"
    reauth(admin)
    result = admin.post(path(asset, "purge"), json={**body, "confirm_path": asset["path"] + "/"})
    assert result.status_code == 422
    response = admin.post(path(asset, "purge"), json=body)
    assert response.status_code == 202, response.text
    receipt = response.json()
    assert receipt["status"] == "PENDING"
    assert admin.post(path(asset, "purge"), json=body).json()["id"] == receipt["id"]
    assert ack(platform, node, receipt["id"], {}).status_code == 409
    command, = platform.commands(node)
    assert command["type"] == PURGE and command["payload"]["identity"] == PIN
    assert admin.get(LEDGER).json()[0]["file_deletion"]["status"] == "RUNNING"
    result = {"asset_id": asset["id"], "path": asset["path"], "status": "DELETED"}
    assert ack(platform, node, receipt["id"], {**result, "path": "/machine/wrong"}).status_code == 422
    assert ack(platform, node, receipt["id"], result).status_code == 200
    assert ack(platform, node, receipt["id"], result).status_code == 200
    assert admin.post(path(asset, "purge"), json=body).json()["id"] == receipt["id"]
    item, = admin.get(LEDGER).json()
    assert item["file_deletion"]["status"] == "DELETED" and item["can_purge"] is False
    assert PIN not in admin.get(LEDGER).text
    with platform.app.state.db.read() as session:
        assert session.get(Asset, asset["id"]).state == "ARCHIVED"


def test_pending_purge_blocks_re_registration_on_another_agent_on_same_machine(team):
    platform, node, _, clients = team
    peer = platform.register()  # The same hostname is another Agent of this machine.
    enable(platform, node)
    admin, operator = clients["team-admin"], clients["alice"]
    asset = registered(team, kind="NORM")
    assert operator.post(archive_path(asset)).status_code == 200
    confirmation = inspect(team, asset, "MISSING")
    reauth(admin)
    receipt = admin.post(path(asset, "purge"), json=confirmation).json()
    assert receipt["status"] == "PENDING"
    for location in [node, peer]:
        response = operator.post("/api/v1/assets", json=body(location, kind="MODEL", path=asset["path"] + "/new"))
        assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_PURGE_PENDING"
    command, = platform.commands(node)
    assert command["payload"]["identity"] is None
    assert ack(platform, node, command["id"], {"asset_id": asset["id"], "path": asset["path"], "status": "MISSING"}).status_code == 200
    response = operator.post("/api/v1/assets", json=body(peer, kind="MODEL", path=asset["path"] + "/new"))
    assert response.status_code == 201, response.text


def test_overlap_and_permission_revocation_stop_purge_before_delivery(team):
    platform, node, users, clients = team
    enable(platform, node)
    admin, operator = clients["team-admin"], clients["alice"]
    asset = registered(team, kind="NORM", path="/machine/old/data")
    assert operator.post(archive_path(asset)).status_code == 200
    parent = registered(team, kind="MODEL", path="/machine/old/./")
    response = admin.post(path(asset, "purge-inspections"))
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_PURGE_PATH_IN_USE"
    assert operator.post(archive_path(parent)).status_code == 200
    confirmation = inspect(team, asset)
    reauth(admin)
    result = admin.post(path(asset, "purge"), json=confirmation)
    assert result.status_code == 202, result.text
    with platform.app.state.db.write() as session:
        from vla_control.auth_models import User
        session.get(User, users["team-admin"]["id"]).role = "operator"
    assert platform.commands(node) == []
    with platform.app.state.db.read() as session:
        command = session.get(Command, result.json()["id"])
        assert command.status == "REJECTED" and command.delivered_at is None
        assert command.result["error"]["code"] == "COMMAND_AUTHORIZATION_REVOKED"


def test_old_archive_cannot_adopt_a_newer_registration_at_the_same_path(team):
    platform, node, _, clients = team
    enable(platform, node)
    admin = clients["team-admin"]
    old = registered(team, kind="NORM")
    assert admin.post(archive_path(old)).status_code == 200
    new = registered(team, kind="NORM", path=old["path"] + "/./")
    assert admin.post(archive_path(new)).status_code == 200
    response = admin.post(path(old, "purge-inspections"))
    assert response.status_code == 409 and response.json()["error"]["code"] == "ASSET_PURGE_PATH_REUSED"
    assert admin.post(path(new, "purge-inspections")).status_code == 202


def test_file_identity_change_needs_new_inspection_and_keeps_old_command(team):
    platform, node, _, clients = team
    enable(platform, node)
    admin = clients["team-admin"]
    asset = registered(team, kind="NORM")
    assert admin.post(archive_path(asset)).status_code == 200
    old = inspect(team, asset)
    reauth(admin)
    response = admin.post(path(asset, "purge"), json=old)
    assert response.status_code == 202, response.text
    command, = platform.commands(node)
    failed = {"error": {"code": "ASSET_PURGE_IDENTITY_CHANGED", "message": "Identity changed"}}
    assert ack(platform, node, command["id"], failed, "REJECTED").status_code == 200
    assert admin.get(LEDGER).json()[0]["requires_new_inspection"] is True
    assert admin.post(path(asset, "purge"), json=old).status_code == 409
    new = inspect(team, asset)
    assert new["inspection_id"] != old["inspection_id"]
    assert admin.post(path(asset, "purge"), json=new).status_code == 202
    with platform.app.state.db.read() as session:
        assert len(list(session.scalars(select(Command).where(Command.type == PURGE)))) == 2


def test_archive_keeps_delivered_validation_visible_as_pending_until_agent_finishes(team):
    platform, node, _, clients = team
    enable(platform, node)
    with platform.app.state.db.write() as session:
        row = session.get(Node, node["node_id"])
        row.metrics = {"capabilities": [INSPECT, PURGE, "DATASET_VALIDATE"]}
        session.get(Profile, (node["node_id"], "default")).adapter = "openpi-sft"
    asset = registered(team)
    validation, = platform.commands(node)
    assert validation["type"] == "DATASET_VALIDATE"
    assert clients["alice"].post(archive_path(asset)).status_code == 200
    admin = clients["team-admin"]
    receipt = admin.post(path(asset, "purge-inspections")).json()
    commands = platform.commands(node)
    assert {command["id"] for command in commands} == {validation["id"], receipt["id"]}
    result = {"asset_id": asset["id"], "path": asset["path"], "status": "READY", "identity": PIN}
    assert ack(platform, node, receipt["id"], result).status_code == 200
    reauth(admin)
    confirmation = {"confirm_path": asset["path"], "inspection_id": receipt["id"]}
    denied = admin.post(path(asset, "purge"), json=confirmation)
    assert denied.status_code == 409 and denied.json()["error"]["code"] == "ASSET_PURGE_NODE_BUSY"
    assert ack(platform, node, validation["id"], {}, "REJECTED").status_code == 200
    assert admin.post(path(asset, "purge"), json=confirmation).status_code == 202


def test_undelivered_purge_expires_without_losing_delivered_receipts(team):
    platform, node, _, clients = team
    enable(platform, node)
    admin = clients["team-admin"]
    asset = registered(team, kind="NORM")
    assert admin.post(archive_path(asset)).status_code == 200
    confirmation = inspect(team, asset)
    reauth(admin)
    response = admin.post(path(asset, "purge"), json=confirmation)
    stale_id = response.json()["id"]
    with platform.app.state.db.write() as session:
        session.get(Command, stale_id).created_at = offset_time(-900)
    assert admin.get(LEDGER).json()[0]["file_deletion"]["status"] == "FAILED"
    assert platform.commands(node) == []
    retry = admin.post(path(asset, "purge"), json=confirmation).json()
    assert retry["id"] != stale_id
    assert platform.commands(node)[0]["id"] == retry["id"]
    with platform.app.state.db.write() as session:
        session.get(Command, retry["id"]).created_at = offset_time(-900)
    assert platform.commands(node)[0]["id"] == retry["id"]
    assert admin.get(LEDGER).json()[0]["file_deletion"]["status"] == "RUNNING"


def test_catalogue_archive_then_actual_agent_purge_in_temporary_directory(team, tmp_path):
    platform, node, _, clients = team
    root = tmp_path.resolve()
    storage = root / "files"
    data = storage / "dataset"
    data.mkdir(parents=True)
    sample = data / "sample.txt"
    sample.write_text("keep until an administrator confirms")
    other = storage / "unrelated"
    other.mkdir()
    outside = other / "sentinel"
    outside.write_text("unrelated asset")
    enable(platform, node)
    with platform.app.state.db.write() as session:
        row = session.get(Node, node["node_id"])
        row.roots, row.output_root = [str(storage)], str(root / "output")
    config = {"roots": [str(storage)], "output_root": str(root / "output"), "state_dir": str(root / "agent"),
              "config_path": str(root / "agent.json"), "profiles": []}
    state = State(config["state_dir"])
    asset = registered(team, path=str(data))
    registered(team, kind="NORM", path=str(outside))
    admin = clients["team-admin"]
    assert clients["alice"].post(archive_path(asset)).status_code == 200
    assert sample.exists() and platform.commands(node) == []
    inspection = admin.post(path(asset, "purge-inspections")).json()
    command, = platform.commands(node)
    assert str(outside) in command["payload"]["protected_paths"]
    scan = execute_asset_purge_command(config, state, INSPECT, command["payload"])
    assert scan["status"] == "READY" and sample.exists()
    assert ack(platform, node, inspection["id"], scan).status_code == 200
    reauth(admin)
    response = admin.post(path(asset, "purge"), json={"confirm_path": str(data), "inspection_id": inspection["id"]})
    assert response.status_code == 202, response.text
    command, = platform.commands(node)
    actual = execute_asset_purge_command(config, state, PURGE, command["payload"])
    assert actual["status"] == "DELETED", actual
    assert ack(platform, node, command["id"], actual).status_code == 200
    assert not data.exists() and outside.read_text() == "unrelated asset"
    assert admin.get(LEDGER).json()[0]["file_deletion"]["status"] == "DELETED"
