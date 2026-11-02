from vla_agent.main import Agent


def test_heartbeat_reports_identity_captured_at_start(agent_fixture, monkeypatch):
    captured = []

    class Client:
        def request(self, method, path, token, body=None, headers=None):
            captured.append(body)
            return {}

    release = {"version": "0.1.0", "build_id": "a" * 64}
    monkeypatch.setattr("vla_agent.main.installed_agent_release", lambda: dict(release))
    monkeypatch.setattr("vla_agent.main.system_metrics", lambda _: {})
    monkeypatch.setattr("vla_agent.main.gpu_inventory", lambda: ([], {}))
    agent = Agent(agent_fixture["config"], client=Client())
    agent.credentials = {"node_id": "fixture-node", "token": "fixture-token"}
    try:
        # Installing new files while this process lives must not change its proof.
        release["build_id"] = "b" * 64
        agent.heartbeat()
        assert captured[-1]["metrics"]["agent"] == {"version": "0.1.0", "build_id": "a" * 64}
    finally:
        agent.close()
