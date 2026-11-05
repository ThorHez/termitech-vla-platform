"""Admin-only audit queries keep filters, cursor boundaries and safe details."""

import base64
import hmac
import json
from uuid import UUID

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete

from vla_control.auth_models import AuditEvent
from vla_control.common import uid


PASSWORD = "isolated-audit-query-password"
STAMP = "2026-11-05T12:00:00.000000+00:00"
ACTOR = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


@pytest.fixture
def audit(platform):
    platform.app.state.auth.create_user("audit-admin", PASSWORD, "admin")
    client = TestClient(platform.app, base_url="https://testserver")
    response = client.post("/api/v1/auth/login", json={"username": "audit-admin", "password": PASSWORD})
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    with platform.app.state.db.write() as session:
        session.execute(delete(AuditEvent))
    try:
        yield platform, client
    finally:
        client.close()


def seed(audit, records):
    with audit[0].app.state.db.write() as session:
        for index, changes in enumerate(records, 1):
            values = {"id": str(UUID(int=index)), "created_at": STAMP, "actor_id": ACTOR,
                "actor_username": "Alice", "action": "assets.register", "method": "POST",
                "path": "/api/v1/assets", "status": 201, "details": {}, "remote_addr": "testclient", **changes}
            session.add(AuditEvent(**values))


def query(audit, **parameters):
    response = audit[1].get("/api/v1/audit/events", params=parameters)
    assert response.status_code == 200, response.text
    return response.json()


def test_keyset_pages_have_stable_ties_and_ignore_a_newer_head(audit):
    seed(audit, [{} for _ in range(6)])
    first = query(audit, limit=2)
    assert [item["id"] for item in first["items"]] == [str(UUID(int=6)), str(UUID(int=5))]
    assert first["has_more"] and first["next_cursor"]
    seed(audit, [{"id": str(UUID(int=9)), "created_at": "2026-11-05T13:00:00.000000+00:00"}])
    second = query(audit, limit=2, cursor=first["next_cursor"])
    third = query(audit, limit=2, cursor=second["next_cursor"])
    assert second["has_more"] and not third["has_more"] and third["next_cursor"] is None
    identifiers = [item["id"] for page in (first, second, third) for item in page["items"]]
    assert identifiers == [str(UUID(int=index)) for index in range(6, 0, -1)]


def test_default_page_size_maximum_and_empty_result_shape(audit):
    assert query(audit) == {"items": [], "next_cursor": None, "has_more": False}
    seed(audit, [{} for _ in range(205)])
    assert len(query(audit)["items"]) == 50
    result = query(audit, limit=200)
    assert len(result["items"]) == 200 and result["has_more"]


def test_filters_combine_and_time_offsets_are_normalized_inclusively(audit):
    base = {"action": "transfers.create", "status": 202, "details": {"target_name": "Cube data"}}
    seed(audit, [base, {**base, "actor_id": OTHER}, {**base, "status": 200},
        {**base, "action": "transfers.retry"}, {**base, "created_at": "2026-11-05T11:59:59.000000+00:00"}])
    result = query(audit, actor_id=ACTOR.upper(), action="transfers.create", outcome="accepted", q="cube",
        from_time="2026-11-05T20:00:00+08:00", to_time="2026-11-05T12:00:00Z")
    assert [item["id"] for item in result["items"]] == [str(UUID(int=1))]
    assert result["items"][0]["outcome"] == "accepted"


def test_unattributed_matches_only_null_not_legacy_actor_text(audit):
    seed(audit, [{"actor_id": None, "actor_username": None}, {"actor_id": "legacy-admin"}, {}])
    assert [item["id"] for item in query(audit, actor_id="unattributed")["items"]] == [str(UUID(int=1))]
    assert query(audit, actor_id=uid())["items"] == []


@pytest.mark.parametrize("outcome,expected", [("succeeded", {200, 201, 204, 302}),
    ("accepted", {202}), ("denied", {401, 403}), ("failed", {400, 404, 409, 422, 429, 499, 500, 503})])
def test_outcomes_distinguish_acceptance_from_completion(audit, outcome, expected):
    seed(audit, [{"status": status} for status in (200, 201, 202, 204, 302, 400, 401, 403, 404, 409, 422, 429, 499, 500, 503)])
    items = query(audit, outcome=outcome)["items"]
    assert {item["status"] for item in items} == expected
    assert all(item["outcome"] == outcome for item in items)


@pytest.mark.parametrize("field", ["action", "actor_username", "actor_id", "path", "operation_label",
                                   "target_id", "target_name", "attempted_username"])
def test_text_search_is_case_insensitive_and_searches_only_supported_fields(audit, field):
    needle = ACTOR if field == "actor_id" else ("needle" if field == "attempted_username" else "Needle")
    values = {field: needle} if field in {"action", "actor_username", "actor_id", "path"} else {"details": {field: needle}}
    seed(audit, [values, {"actor_id": OTHER, "actor_username": "Bob", "details": {"reason": needle}}])
    assert [item["id"] for item in query(audit, q=needle.lower())["items"]] == [str(UUID(int=1))]


@pytest.mark.parametrize("needle", ["%", "_", "\\", "' OR 1=1 --"])
def test_search_treats_sql_wildcards_and_sql_text_literally(audit, needle):
    seed(audit, [{"details": {"target_name": "prefix " + needle + " suffix"}}, {"details": {"target_name": "anything"}}])
    assert [item["id"] for item in query(audit, q=needle)["items"]] == [str(UUID(int=1))]


@pytest.mark.parametrize("parameters", [{"limit": 0}, {"limit": 201}, {"limit": "not-a-number"},
    {"actor_id": "not-a-uuid"}, {"actor_id": ""}, {"action": ""}, {"action": "a" * 121},
    {"q": "q" * 101}, {"q": "bad\nquery"}, {"outcome": "completed"},
    {"from_time": "2026-11-05T12:00:00"}, {"to_time": "2026-99-01T00:00:00Z"},
    {"from_time": "2026-11-06T00:00:00Z", "to_time": "2026-11-05T00:00:00Z"}])
def test_invalid_filters_fail_closed(audit, parameters):
    assert audit[1].get("/api/v1/audit/events", params=parameters).status_code == 422


@pytest.mark.parametrize("changed", [{"limit": 2}, {"actor_id": ACTOR}, {"action": "assets.register"},
    {"outcome": "succeeded"}, {"q": "asset"}, {"from_time": "2026-11-01T00:00:00Z"},
    {"to_time": "2026-11-06T00:00:00Z"}])
def test_cursors_are_bound_to_all_filters_and_page_size(audit, changed):
    seed(audit, [{}, {}, {}])
    first = query(audit, limit=1)
    response = audit[1].get("/api/v1/audit/events", params={"limit": 1, "cursor": first["next_cursor"], **changed})
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_AUDIT_CURSOR"


@pytest.mark.parametrize("corruption", ["tamper", "invalid-base64", "unknown-key", "version", "id", "time", "duplicate-key"])
def test_cursor_signature_and_payload_structure_are_checked(audit, corruption):
    seed(audit, [{}, {}])
    cursor = query(audit, limit=1)["next_cursor"]
    if corruption == "tamper":
        cursor = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")
    elif corruption == "invalid-base64":
        cursor = "not!base64.bad"
    else:
        encoded = cursor.split(".")[0]
        value = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if corruption == "unknown-key":
            value["extra"] = "not-allowed"
        elif corruption == "version":
            value["v"] = True
        elif corruption == "id":
            value["id"] = "not-a-uuid"
        elif corruption == "time":
            value["at"] = "2026-11-05T12:00:00"
        payload = json.dumps(value, separators=(",", ":")).encode()
        if corruption == "duplicate-key":
            payload = payload[:-1] + b',"v":1}'
        settings = audit[0].settings
        key = hmac.digest(settings.enrollment_token.encode(), b"vla-audit-cursor-v1", "sha256")
        encode = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
        cursor = encode(payload) + "." + encode(hmac.digest(key, payload, "sha256"))
    response = audit[1].get("/api/v1/audit/events", params={"limit": 1, "cursor": cursor})
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_AUDIT_CURSOR"


def test_legacy_array_api_and_new_serializer_filter_sensitive_unknown_details(audit):
    secret = "never-expose-audit-body-secret"
    seed(audit, [{"details": {"operation_label": "登记资产", "category": "assets", "target_id": OTHER,
        "target_name": "Safe name", "duration_ms": 3, "asset_kind": "DATASET", "related_ids": OTHER,
        "counts": {"created": 2, "password": secret, "failed": -1}, "password": secret,
        "token": secret, "body": {"secret": secret}, "query": secret, "terminal_input": secret}}])
    current = query(audit)["items"]
    previous = audit[1].get("/api/v1/audit", params={"limit": 500})
    assert previous.status_code == 200 and isinstance(previous.json(), list)
    assert "outcome" not in previous.json()[0] and current[0]["outcome"] == "succeeded"
    for response in (previous.json()[0], current[0]):
        assert response["details"]["counts"] == {"created": 2}
        assert response["details"]["related_ids"] == OTHER
        assert secret not in json.dumps(response)
        assert not {"password", "token", "body", "query", "terminal_input"} & response["details"].keys()
    assert audit[1].get("/api/v1/audit", params={"limit": 501}).status_code == 422


@pytest.mark.parametrize("role", ["operator", "viewer"])
def test_audit_queries_remain_admin_only(audit, role):
    platform, _ = audit
    platform.app.state.auth.create_user("audit-" + role, PASSWORD, role)
    client = TestClient(platform.app, base_url="https://testserver")
    try:
        assert client.get("/api/v1/audit/events").status_code == 401
        login = client.post("/api/v1/auth/login", json={"username": "audit-" + role, "password": PASSWORD})
        assert login.status_code == 200
        assert client.get("/api/v1/audit/events").status_code == 403
        assert client.get("/api/v1/audit").status_code == 403
    finally:
        client.close()
