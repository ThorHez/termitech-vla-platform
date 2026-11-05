"""Auditing preserves transport and rejects oversized or private response data."""

import json
from types import SimpleNamespace

import anyio
import pytest

from vla_control.authorization import AUTH_ROUTES, PERMISSIONS
from vla_control.operation_audit import MAX_RESPONSE_BYTES, OPERATIONS, OperationAuditMiddleware, context_details


class AuditSink:
    def __init__(self, *, fail=False):
        self.entries = []
        self.fail = fail

    def audit(self, request, action, status, details):
        if self.fail:
            raise RuntimeError("private-storage-error-do-not-disclose")
        self.entries.append({"action": action, "status": status, "details": details})


def exercise(*, path="/api/v1/assets", method="POST", status=201, chunks=(), headers=None,
             committed=False, fail_storage=False, target_name=None):
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "scheme": "https", "method": method, "path": path, "raw_path": path.encode(),
             "root_path": "", "query_string": b"token=never-copy-queries", "headers": [],
             "server": ("testserver", 443), "client": ("192.0.2.1", 1234), "state": {}}
    messages, received = [], []
    sink = AuditSink(fail=fail_storage)

    async def application(actual_scope, _receive, send):
        actual_scope["route"] = SimpleNamespace(path=path)
        if committed:
            actual_scope["state"]["audit_recorded_id"] = "committed-explicit-event"
        if target_name:
            actual_scope["state"]["operation_target_name"] = target_name
            actual_scope["path_params"] = {"asset_id": "4a98105e-cb11-403d-83f8-4d467ea8097b"}
        await send({"type": "http.response.start", "status": status,
                    "headers": headers or [(b"content-type", b"application/json")]})
        for number, chunk in enumerate(chunks):
            await send({"type": "http.response.body", "body": chunk,
                        "more_body": number < len(chunks) - 1})

    async def receive():
        received.append(True)
        raise AssertionError("The audit observer must not read request bodies")

    async def send(message):
        messages.append(dict(message))

    async def run():
        await OperationAuditMiddleware(application, sink)(scope, receive, send)

    anyio.run(run)
    assert received == []
    return sink.entries, messages


def test_every_registered_human_mutation_has_a_semantic_operation():
    required = {key for key in set(PERMISSIONS) | AUTH_ROUTES
                if key[0] not in {"GET", "HEAD", "OPTIONS"} and key[1] != "/api/v1/auth/activity"}
    assert required <= set(OPERATIONS)
    assert all(OPERATIONS[key].action != "api.mutation" and OPERATIONS[key].label for key in required)


def test_local_console_keeps_target_identity_and_session_revocation_semantics():
    details = context_details(None, "auth.sessions_revoked", {"target_user_id": "user-id",
        "target_username": "local-user", "count": 2})
    assert details["target_id"] == "user-id" and details["target_name"] == "local-user"
    assert details["operation_label"] == "撤销登录会话" and details["category"] == "auth"


@pytest.mark.parametrize("status", [200, 201, 202, 401, 403, 409, 422, 500])
def test_response_stream_status_headers_and_safe_projection_remain_exact(status):
    payload = {"id": "4a98105e-cb11-403d-83f8-4d467ea8097b", "name": "safe resource",
        "kind": "MODEL", "password": "private-password", "csrf_token": "private-csrf", "ticket": "private-ticket",
        "probe": {"stdout": "private-remote-output"},
        "error": {"code": "SAFE_ERROR", "message": "private-error-message"}}
    raw = json.dumps(payload).encode()
    chunks = [raw[:15], raw[15:88], raw[88:]]
    headers = [(b"content-type", b"application/json"), (b"set-cookie", b"private-cookie")]
    entries, messages = exercise(status=status, chunks=chunks, headers=headers)
    assert messages[0] == {"type": "http.response.start", "status": status, "headers": headers}
    assert [message["body"] for message in messages[1:]] == chunks
    assert [message["more_body"] for message in messages[1:]] == [True, True, False]
    assert len(entries) == 1 and entries[0]["status"] == status and entries[0]["action"] == "assets.register"
    details = entries[0]["details"]
    assert details["operation_label"] == "登记资产" and details["request_id"] and details["duration_ms"] >= 0
    if status < 400:
        assert details["target_id"] == payload["id"] and details["target_name"] == "safe resource"
        assert details["asset_kind"] == "MODEL"
    else:
        assert details["error_code"] == "SAFE_ERROR" and "target_name" not in details
    encoded = json.dumps(entries)
    for secret in ("private-password", "private-csrf", "private-ticket", "private-remote-output",
                   "private-error-message", "private-cookie", "never-copy-queries"):
        assert secret not in encoded


def test_oversized_response_is_forwarded_without_loss_and_keeps_preloaded_object_identity():
    data = b'{"password":"' + b"private-data" * (MAX_RESPONSE_BYTES // 5) + b'"}'
    entries, messages = exercise(path="/api/v1/assets/{asset_id}/archive", chunks=[data[:200], data[200:]],
                                 target_name="large dataset")
    assert b"".join(message["body"] for message in messages[1:]) == data
    assert entries[0]["details"]["target_name"] == "large dataset"
    assert entries[0]["details"]["target_id"] == "4a98105e-cb11-403d-83f8-4d467ea8097b"
    assert "private-data" not in json.dumps(entries)


@pytest.mark.parametrize("path,method,status", [
    ("/api/v1/assets", "GET", 200), ("/api/v1/audit/events", "GET", 200),
    ("/api/v1/auth/me", "GET", 401), ("/api/v1/auth/activity", "POST", 200),
    ("/api/v1/auth/activity", "POST", 401), ("/agent/v1/heartbeat", "POST", 200),
    ("/api/v1/assets", "POST", 307),
])
def test_read_polls_activity_and_agent_protocol_do_not_generate_user_operations(path, method, status):
    entries, _ = exercise(path=path, method=method, status=status, chunks=[b"{}"])
    assert entries == []


def test_explicit_committed_record_suppresses_only_the_http_fallback():
    entries, _ = exercise(chunks=[b"{}"], committed=True)
    assert entries == []


def test_audit_storage_error_cannot_change_an_accepted_business_response(caplog):
    entries, messages = exercise(chunks=[b'{"id":"created-resource"}'], status=201, fail_storage=True)
    assert entries == [] and messages[0]["status"] == 201
    assert messages[-1]["body"] == b'{"id":"created-resource"}'
    assert "Unable to persist operation audit" in caplog.text
    assert "private-storage-error" not in caplog.text


def test_unexpected_metadata_types_cannot_break_the_business_response():
    entries, messages = exercise(chunks=[b'{"kind":{"token":"private-value"},"name":[],"id":42}'])
    assert messages[0]["status"] == 201 and len(entries) == 1
    assert "private-value" not in json.dumps(entries)


def test_explicit_binary_download_is_audited_without_capturing_binary_bytes():
    entries, messages = exercise(path="/api/v1/agent-package", method="GET", status=200,
        headers=[(b"content-type", b"application/gzip")], chunks=[b"private-binary-content"])
    assert messages[-1]["body"] == b"private-binary-content"
    assert entries[0]["action"] == "machines.download_agent"
    assert "private-binary-content" not in json.dumps(entries)
