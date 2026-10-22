"""Offline session invalidation preserves users and machine-local metadata."""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from vla_control.auth_models import AuditEvent, User, UserSession
from vla_control.common import now, token_hash
from vla_control.main import create_app
from vla_control.models import Asset, Node, Profile
from vla_control.settings import Settings
from vla_control.users import main as users_cli


PASSWORD = "isolated-session-revocation-test"


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.delenv("VLA_DATABASE_URL", raising=False)

    def no_password_prompt(_):
        pytest.fail("Session revocation must never ask for a password")

    monkeypatch.setattr("vla_control.users.getpass.getpass", no_password_prompt)
    settings = Settings(database_url="sqlite:///" + str(tmp_path / "accounts.sqlite"),
                        enrollment_token="agents-separate", scheduler_enabled=False, session_cookie_secure=False)
    app = create_app(settings)
    app.state.auth.create_user("admin", PASSWORD, "admin")
    app.state.auth.create_user("viewer", PASSWORD, "viewer")
    with app.state.db.write() as session:
        session.add(Node(id="retained-node", name="retained", hostname="retained", token_hash=token_hash("node-secret"),
                         status="OFFLINE", last_seen=now(), created_at=now(), roots=["/machine"], output_root="/machine/out"))
        session.flush()
        session.add(Profile(node_id="retained-node", id="retained-profile", name="retained-runtime", code_dir="/machine/code",
                            python="/machine/python", adapter="smoke", recipe="smoke"))
        session.add(Asset(id="retained-asset", node_id="retained-node", name="retained-data", kind="dataset",
                          path="/machine/data", version="v1", created_at=now()))
    admin = TestClient(app)
    viewer = TestClient(app)
    for client, username in ((admin, "admin"), (viewer, "viewer")):
        response = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})
        assert response.status_code == 200, response.text
    yield app, settings.database_url, admin, viewer
    admin.close()
    viewer.close()
    app.state.remote_manager.close()
    app.state.db.close()


def retained_state(app):
    with app.state.db.read() as session:
        return {table.name: [dict(row) for row in session.execute(select(table)).mappings()]
                for table in (User.__table__, Node.__table__, Profile.__table__, Asset.__table__)}


def test_revoke_all_invalidates_old_cookies_and_preserves_metadata(accounts, capsys):
    app, url, admin, viewer = accounts
    before = retained_state(app)
    users_cli(["revoke-sessions", "--all", "--database-url", url])
    assert admin.get("/api/v1/auth/me").status_code == 401
    assert viewer.get("/api/v1/auth/me").status_code == 401
    assert retained_state(app) == before
    output = capsys.readouterr().out
    assert "2" in output and PASSWORD not in output and "node-secret" not in output
    with app.state.db.read() as session:
        assert session.scalar(select(UserSession)) is None
        event = session.scalar(select(AuditEvent).where(AuditEvent.action == "auth.sessions_revoked"))
        assert event.method == "CLI" and event.path == "local-console" and event.status == 200
        assert event.details == {"count": 2}
    # Revocation leaves the same password usable for a new independent session.
    response = admin.post("/api/v1/auth/login", json={"username": "admin", "password": PASSWORD})
    assert response.status_code == 200 and response.json()["user"]["role"] == "admin"


def test_revoke_one_account_preserves_other_sessions(accounts, capsys):
    app, url, admin, viewer = accounts
    before = retained_state(app)
    users_cli(["revoke-sessions", "--username", " ADMIN ", "--database-url", url])
    assert admin.get("/api/v1/auth/me").status_code == 401
    assert viewer.get("/api/v1/auth/me").status_code == 200
    assert retained_state(app) == before
    output = capsys.readouterr().out
    assert "1" in output and "admin" in output and PASSWORD not in output
    with app.state.db.read() as session:
        event = session.scalar(select(AuditEvent).where(AuditEvent.action == "auth.sessions_revoked"))
        assert event.details["target_username"] == "admin" and event.details["count"] == 1


def test_missing_user_is_404_and_keeps_all_sessions(accounts, capsys):
    app, url, admin, viewer = accounts
    before = retained_state(app)
    with pytest.raises(SystemExit) as exc:
        users_cli(["revoke-sessions", "--username", "missing", "--database-url", url])
    assert exc.value.code == 2 and "NOT_FOUND" in capsys.readouterr().err
    assert retained_state(app) == before
    assert admin.get("/api/v1/auth/me").status_code == viewer.get("/api/v1/auth/me").status_code == 200
    with app.state.db.read() as session:
        assert session.scalar(select(AuditEvent).where(AuditEvent.action == "auth.sessions_revoked")) is None


@pytest.mark.parametrize("arguments", [[], ["--all", "--username", "admin"]])
def test_revoke_cli_requires_one_explicit_selector(arguments, monkeypatch):
    monkeypatch.delenv("VLA_DATABASE_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        users_cli(["revoke-sessions", *arguments])
    assert exc.value.code == 2


def test_revoke_cli_requires_explicit_database(monkeypatch):
    monkeypatch.delenv("VLA_DATABASE_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        users_cli(["revoke-sessions", "--all"])
    assert exc.value.code == 2


def test_revoke_cli_accepts_explicit_database_environment(accounts, monkeypatch):
    _, url, admin, viewer = accounts
    monkeypatch.setenv("VLA_DATABASE_URL", url)
    users_cli(["revoke-sessions", "--all"])
    assert admin.get("/api/v1/auth/me").status_code == viewer.get("/api/v1/auth/me").status_code == 401


def test_revoke_without_existing_sessions_is_repeatable(accounts, capsys):
    app, url, _, _ = accounts
    users_cli(["revoke-sessions", "--all", "--database-url", url])
    capsys.readouterr()
    users_cli(["revoke-sessions", "--all", "--database-url", url])
    assert "0" in capsys.readouterr().out
    with app.state.db.read() as session:
        events = session.scalars(select(AuditEvent).where(AuditEvent.action == "auth.sessions_revoked")
                                 .order_by(AuditEvent.created_at)).all()
        assert [event.details["count"] for event in events] == [2, 0]
