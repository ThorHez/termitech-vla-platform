from concurrent.futures import ThreadPoolExecutor

import pytest

from vla_agent.main import Agent
from vla_agent.state import State


def test_event_spool_transactional_order_survives_reopen(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    state = State(config["state_dir"])
    assert state.create_attempt(payload, config["output_root"] + "/one")
    assert not state.create_attempt(payload, config["output_root"] + "/one")
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(lambda number: state.emit(payload["attempt_id"], "METRIC", {"value": number}), range(30)))
    reopened = State(config["state_dir"])
    events = reopened.pending_events()
    assert sorted(event["seq"] for event in events) == list(range(1, 31))
    reopened.acknowledge_events(events[:10])
    assert len(reopened.pending_events()) == 20
    reopened.emit(payload["attempt_id"], "FAILED", {"exit_code": 1})
    reopened.emit(payload["attempt_id"], "SUCCEEDED", {"exit_code": 0})
    assert reopened.attempt(payload["attempt_id"])["state"] == "FAILED"


class FakeClient:
    def __init__(self):
        self.registrations = 0
        self.accept = False

    def request(self, method, path, token, body=None, headers=None):
        if path.endswith("register"):
            assert headers.get("Idempotency-Key")
            self.registrations += 1
            return {"node_id": "node-1", "token": "private-node-token"}
        if path.endswith("events/batch"):
            if not self.accept:
                raise OSError("network unavailable")
            return {"accepted": len(body["events"])}
        return {"ok": True}


def test_credentials_and_pending_events_survive_network_outage(agent_fixture):
    client = FakeClient()
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    agent = Agent(config, client)
    agent.register()
    agent.state.create_attempt(payload, config["output_root"] + "/one")
    agent.state.emit(payload["attempt_id"], "LOG", {"text": "persist me"})
    with pytest.raises(OSError):
        agent.flush_events()
    assert len(agent.state.pending_events()) == 1
    restarted = Agent(config, client)
    restarted.register()
    assert client.registrations == 1
    client.accept = True
    restarted.flush_events()
    assert restarted.state.pending_events() == []


def test_cancel_token_is_bound_to_attempt(agent_fixture):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    state = State(config["state_dir"])
    state.create_attempt(payload, config["output_root"])
    with pytest.raises(ValueError, match="token mismatch"):
        state.request_cancel(payload["attempt_id"], "wrong-token")
    assert state.attempt(payload["attempt_id"])["cancel_requested"] == 0
