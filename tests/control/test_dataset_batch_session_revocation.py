"""A queued HTTP worker cannot register assets with a revoked browser session."""

import pytest
from sqlalchemy import delete, select

from vla_control.auth_models import UserSession
from vla_control.common import offset_time
from vla_control import dataset_batches as module

from .test_dataset_batch_security import base, body, counts, login, ready, secured  # noqa: F401


@pytest.mark.parametrize("operation", ["scan", "register"])
@pytest.mark.parametrize("change", ["disabled", "demoted", "logout", "idle", "expired"])
@pytest.mark.parametrize("username", ["batch-owner", "batch-operator"])
def test_batch_worker_rechecks_original_session_after_final_http_authorization(secured, monkeypatch, operation, change, username):
    authentication = login(secured, username)
    user_id = authentication["user"]["id"]
    scan = ready(secured) if operation == "register" else None
    before = counts(secured)
    app = secured[0].app
    original = module.run_in_threadpool
    invoked = []
    async def revoke_before_worker(function, *args, **kwargs):
        invoked.append(True)
        if change in {"disabled", "demoted"}:
            app.state.auth.update_user(None, user_id,
                {"enabled": False} if change == "disabled" else {"role": "viewer"})
        else:
            with app.state.db.write() as session:
                if change == "logout":
                    session.execute(delete(UserSession).where(UserSession.user_id == user_id))
                else:
                    row = session.scalar(select(UserSession).where(UserSession.user_id == user_id))
                    if change == "idle":
                        row.last_activity_at = offset_time(-3601)
                    else:
                        row.expires_at = offset_time(-1)
        return await original(function, *args, **kwargs)
    monkeypatch.setattr(module, "run_in_threadpool", revoke_before_worker)
    path = base(secured) + (f"/{scan['id']}/register" if scan is not None else "")
    response = secured[1].post(path, json=body() if operation == "register" else {"path": "/machine/datasets"})
    assert invoked and response.status_code in {401, 403}, response.text
    assert counts(secured) == before
