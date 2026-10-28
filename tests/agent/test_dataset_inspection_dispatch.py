"""Dataset reads use the durable background command worker, never training."""

import time
from types import SimpleNamespace

from vla_agent import dataset_inspection as inspector
from vla_agent.main import Agent, _execute_preflight_command
from vla_agent.state import State
from test_background_preflight import CommandClient, connected_agent


def test_dataset_probe_dispatch_is_durable_and_replayed_without_reread(agent_fixture, monkeypatch):
    config = agent_fixture['config'];calls = []
    command = {'id': 'metadata-query', 'type': 'DATASET_INSPECT', 'payload': {'path': '/dataset'}}
    monkeypatch.setattr('vla_agent.main.os.setsid', lambda: None)
    monkeypatch.setattr('vla_agent.main.signal.signal', lambda *_: None)
    def inspect(configured, payload):
        calls.append((configured, payload))
        return {'path': payload['path'], 'schema_version': 1}
    monkeypatch.setattr(inspector, 'inspect_dataset', inspect)
    _execute_preflight_command(config, command)
    _execute_preflight_command(config, command)
    assert len(calls) == 1 and calls[0] == (config, command['payload'])
    assert State(config['state_dir']).command(command['id']) == {
        'status': 'APPLIED', 'result': {'path': '/dataset', 'schema_version': 1}}


def test_metadata_read_does_not_enter_training_path_or_block_cancel_heartbeat(agent_fixture, monkeypatch):
    command = {'id': 'metadata-query', 'type': 'DATASET_INSPECT', 'payload': {'path': '/dataset'}}
    payload = agent_fixture['payload']
    cancel = {'id': 'cancel-now', 'type': 'CANCEL', 'payload': {
        'attempt_id': payload['attempt_id'], 'execution_token': payload['execution_token']}}
    client = CommandClient([command, cancel]);agent = connected_agent(agent_fixture, client)
    agent.state.create_attempt(payload, str(agent_fixture['root'] / 'out'))
    calls = []
    class Process:
        exitcode = None
        def __init__(self, **kwargs): calls.append(kwargs)
        def start(self): pass
        def is_alive(self): return True
    monkeypatch.setattr('vla_agent.main.multiprocessing.get_context', lambda *_: SimpleNamespace(Process=Process))
    monkeypatch.setattr('vla_agent.main.run_preflight', lambda *_: (_ for _ in ()).throw(AssertionError('training was invoked')))
    monkeypatch.setattr('vla_agent.main.gpu_inventory', lambda: ([], {'available': False}))
    monkeypatch.setattr('vla_agent.main.system_metrics', lambda _: {})
    monkeypatch.setattr(agent.runtime_monitor, 'snapshot', lambda _: [])
    try:
        start = time.monotonic()
        agent.poll_commands();agent.poll_commands();agent.heartbeat()
        assert time.monotonic() - start < 1
        assert len(calls) == 1 and calls[0]['args'][1] == command
        assert 'metadata-query' not in client.acks and 'cancel-now' in client.acks
        assert len(client.heartbeats) == 1
        assert agent.state.attempt(payload['attempt_id'])['cancel_requested'] == 1
    finally:
        agent._preflight_worker = None
        agent.close()
