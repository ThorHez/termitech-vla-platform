"""Terminal lifecycle audit attributes attempts without recording shell data."""

import json
import time

import pytest
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from vla_control.auth_models import AuditEvent
from vla_control.common import APIError
from vla_control.ssh_remote import RemoteError

from .test_node_terminals import ORIGIN, PASSWORD, authenticate, terminal, ticket


META = {'operation_label', 'category', 'target_type', 'target_id', 'request_id', 'duration_ms'}


def core(row):
    return {key: value for key, value in row.details.items() if key not in META}


def events(terminal, count=1):
    deadline = time.monotonic() + 2
    while True:
        with terminal.db.read() as session:
            result = list(session.scalars(select(AuditEvent).where(AuditEvent.action.like('terminal.%'))
                                          .order_by(AuditEvent.created_at, AuditEvent.id)))
        if len(result) >= count or time.monotonic() >= deadline:
            return result
        time.sleep(.01)


def test_open_and_close_have_one_real_actor_each_without_ssh_identity_ticket_or_shell_data(terminal):
    issued = ticket(terminal)
    secret = 'terminal-audit-private-shell-input'
    with terminal.client.websocket_connect(issued['websocket_path'], headers=ORIGIN) as websocket:
        authenticate(websocket, issued)
        assert websocket.receive_json()['type'] == 'ready'
        assert websocket.receive_bytes()
        websocket.send_json({'type': 'input', 'data': secret})
        assert websocket.receive_bytes() == secret.encode()
    rows = events(terminal, 2)
    assert [row.action for row in rows] == ['terminal.open', 'terminal.close']
    assert all(row.actor_id == terminal.user_id and row.actor_username == 'admin' and row.status == 200 for row in rows)
    assert all(row.details['node_id'] == terminal.node_id and row.details['terminal_id'] == issued['id'] for row in rows)
    assert all(set(row.details) <= {'node_id', 'terminal_id', 'code'} | META for row in rows)
    saved = json.dumps([{'path': row.path, 'details': row.details} for row in rows])
    assert all(value not in saved for value in (secret, issued['ticket'], PASSWORD, 'fixture-ssh-user'))


@pytest.mark.parametrize('failure,status,code', [
    (RemoteError('SSH_AUTH_FAILED', 'fixture private SSH failure text'), 502, 'SSH_AUTH_FAILED'),
    (TimeoutError('fixture private timeout text'), 504, 'TERMINAL_TIMEOUT'),
    (APIError(403, 'TERMINAL_ACCOUNT_REVOKED', 'fixture private account failure text'), 403, 'TERMINAL_ACCOUNT_REVOKED'),
])
def test_authenticated_ssh_open_failures_are_not_recorded_as_successful_closes(terminal, monkeypatch, failure, status, code):
    issued = ticket(terminal)

    def failed_open(*_):
        raise failure

    monkeypatch.setattr(terminal.manager, '_open', failed_open)
    with terminal.client.websocket_connect(issued['websocket_path'], headers=ORIGIN) as websocket:
        authenticate(websocket, issued)
        assert websocket.receive_json()['code'] == code
        assert websocket.receive_json()['type'] == 'closed'
    rows = events(terminal)
    assert len(rows) == 1
    row = rows[0]
    assert row.action == 'terminal.open_failed' and row.status == status and row.actor_id == terminal.user_id
    assert core(row) == {'node_id': terminal.node_id, 'terminal_id': issued['id'], 'code': code}
    assert 'private' not in json.dumps(row.details) and issued['ticket'] not in json.dumps(row.details)


def test_authenticated_wrong_ticket_is_attributed_without_echoing_the_ticket(terminal):
    issued = ticket(terminal)
    with terminal.client.websocket_connect(issued['websocket_path'], headers=ORIGIN) as websocket:
        authenticate(websocket, issued, ticket='private-wrong-ticket')
        assert websocket.receive_json()['code'] == 'TERMINAL_TICKET_INVALID'
        assert websocket.receive_json()['type'] == 'closed'
    rows = events(terminal)
    assert len(rows) == 1 and rows[0].action == 'terminal.open_failed' and rows[0].status == 403
    assert rows[0].actor_id == terminal.user_id
    assert core(rows[0]) == {'node_id': terminal.node_id, 'terminal_id': issued['id'], 'code': 'TERMINAL_TICKET_INVALID'}


def test_authenticated_disconnect_before_auth_is_a_cancelled_open_without_fake_close(terminal):
    issued = ticket(terminal)
    with terminal.client.websocket_connect(issued['websocket_path'], headers=ORIGIN):
        pass
    rows = events(terminal)
    assert len(rows) == 1 and rows[0].action == 'terminal.open_failed' and rows[0].status == 499
    assert rows[0].actor_id == terminal.user_id and rows[0].details['code'] == 'TERMINAL_DISCONNECTED'


@pytest.mark.parametrize('failure', ['origin', 'anonymous'])
def test_failures_before_identity_verification_do_not_guess_an_actor(terminal, failure):
    issued = ticket(terminal)
    headers = ORIGIN
    if failure == 'origin':
        headers = {'origin': 'https://untrusted.invalid'}
    else:
        terminal.client.cookies.clear()
    with pytest.raises(WebSocketDisconnect):
        with terminal.client.websocket_connect(issued['websocket_path'], headers=headers):
            pass
    with terminal.db.read() as session:
        assert not list(session.scalars(select(AuditEvent).where(AuditEvent.action.like('terminal.%'))))
