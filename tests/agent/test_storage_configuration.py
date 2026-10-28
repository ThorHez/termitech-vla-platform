import copy
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path

import pytest

from vla_agent import storage_configuration as storage
from vla_agent.artifacts import ArtifactMonitor
from vla_agent.config import load_config
from vla_agent.main import Agent
from vla_agent.profile_rebinding import profile_configuration_lock
from vla_agent.recipe_writeback import checkout_lock
from vla_agent.safety import AgentError
from vla_agent.state import State


@pytest.fixture
def storage_fixture(agent_fixture):
    config = agent_fixture['config']
    path = Path(config['config_path'])
    disk = json.loads(path.read_text())
    disk['unrelated_setting'] = {'credential': 'keep-this-private', 'nested': [1, 2]}
    path.write_text(json.dumps(disk))
    state = State(config['state_dir'])
    state.save_credentials({'node_id': 'node-1', 'token': 'keep-node-token'})
    request = {'expected_output_root': config['output_root'], 'expected_cache_root': None,
               'output_root': str(agent_fixture['root'] / 'large-disk/outputs'),
               'cache_root': str(agent_fixture['root'] / 'large-disk/cache'),
               'expires_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}
    return {**agent_fixture, 'state': state, 'request': request, 'path': path}


def configure(fixture, **changes):
    return storage.configure_storage(fixture['config'], fixture['state'], {**fixture['request'], **changes},
                                     operation_id='storage-1')


def test_persists_both_paths_and_preserves_source_private_configuration(storage_fixture):
    fixture = storage_fixture
    old = fixture['path'].read_bytes()
    marker = Path(fixture['config']['output_root']) / 'old-checkpoint'
    marker.write_text('keep old output')
    result = configure(fixture)
    assert result['persisted'] is True
    assert result['output_root'] == fixture['request']['output_root']
    assert result['cache_root'] == fixture['request']['cache_root']
    for entry in result['directories'].values():
        assert entry['free_bytes'] >= 0 and entry['total_bytes'] >= entry['free_bytes']
        assert Path(entry['path']).is_dir()
        assert not list(Path(entry['path']).glob('.vla-storage-probe-*'))
    disk = json.loads(fixture['path'].read_text())
    original = json.loads(old)
    for key in ('profiles', 'state_dir', 'enrollment_token', 'unrelated_setting', 'roots'):
        assert disk[key] == original[key]
    assert fixture['state'].credentials()['token'] == 'keep-node-token'
    assert marker.read_text() == 'keep old output'
    assert disk[storage.HISTORY_KEY] == [original['output_root']]
    assert len(disk[storage.OPERATIONS_KEY]) == 1
    assert Path(storage.cache_environment(fixture['config'])['HF_DATASETS_CACHE']).is_dir()
    backups = list((Path(fixture['config']['state_dir']) / 'profile-rebind').glob('before-storage-*.json'))
    assert len(backups) == 1 and backups[0].read_bytes() == old


def test_saved_operation_replays_after_restart_even_after_its_deadline(storage_fixture, monkeypatch):
    fixture = storage_fixture
    first = configure(fixture)
    persisted = fixture['path'].read_bytes()
    fixture['config'] = load_config(fixture['path'])
    monkeypatch.setattr(storage, '_probe_directory', lambda *a: pytest.fail('replay must not reprobe'))
    assert configure(fixture) == first
    assert fixture['path'].read_bytes() == persisted
    assert fixture['config']['output_root'] == first['output_root']


def test_replay_recovers_a_durable_write_before_live_memory_was_updated(storage_fixture, monkeypatch):
    fixture = storage_fixture
    persist = storage._persist

    def interrupted(*args, **kwargs):
        persist(*args, **kwargs)
        raise storage.StorageConfigurationError('STORAGE_CONFIGURE_RECOVERY_PENDING', 'fixture interrupted', persisted=None)

    monkeypatch.setattr(storage, '_persist', interrupted)
    original_root = fixture['config']['output_root']
    with pytest.raises(AgentError) as exc:
        configure(fixture)
    assert exc.value.code == 'STORAGE_CONFIGURE_RECOVERY_PENDING'
    assert fixture['config']['output_root'] == original_root
    assert json.loads(fixture['path'].read_text())['output_root'] == fixture['request']['output_root']
    recovered = configure(fixture)
    assert recovered['persisted'] is True
    assert fixture['config']['output_root'] == fixture['request']['output_root']
    assert fixture['config']['cache_root'] == fixture['request']['cache_root']


def test_same_operation_id_cannot_be_reused_with_a_different_target(storage_fixture):
    fixture = storage_fixture
    configure(fixture)
    with pytest.raises(AgentError) as exc:
        configure(fixture, cache_root=None)
    assert exc.value.code == 'STORAGE_CONFIGURE_RECOVERY_PENDING'
    assert 'persisted' not in exc.value.as_dict()


@pytest.mark.parametrize('field,value', [('output_root', 'relative'), ('cache_root', ''), ('cache_root', '/'),
                                         ('output_root', '/tmp/../bad'), ('cache_root', '/tmp/a\nb'),
                                         ('expires_at', '2026-10-27T12:00:00'), ('output_root', None)])
def test_invalid_inputs_leave_file_and_live_settings_intact(storage_fixture, field, value):
    fixture = storage_fixture
    old = fixture['path'].read_bytes()
    live = copy.deepcopy(fixture['config'])
    with pytest.raises(AgentError) as exc:
        configure(fixture, **{field: value})
    assert exc.value.code == 'STORAGE_CONFIGURE_INVALID'
    assert exc.value.as_dict()['persisted'] is False
    assert fixture['path'].read_bytes() == old and fixture['config'] == live


def test_directory_alias_and_escaping_symlink_are_not_saved(storage_fixture, tmp_path):
    fixture = storage_fixture
    target = fixture['root'] / 'actual'
    target.mkdir()
    alias = fixture['root'] / 'alias'
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(AgentError) as exc:
        configure(fixture, output_root=str(alias))
    assert exc.value.code == 'STORAGE_CONFIGURE_INVALID'
    outside = fixture['root'].parent / 'outside-storage'
    outside.mkdir(exist_ok=True)
    escape = fixture['root'] / 'escape'
    escape.symlink_to(outside, target_is_directory=True)
    with pytest.raises(AgentError) as exc:
        configure(fixture, cache_root=str(escape))
    assert exc.value.code == 'STORAGE_CONFIGURE_INVALID'


def test_cas_expiry_and_busy_tasks_are_rejected(storage_fixture):
    fixture = storage_fixture
    for change, code in [({'expected_cache_root': str(fixture['root'] / 'another-cache')}, 'CONFLICT'),
                         ({'expires_at': (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()}, 'EXPIRED')]:
        with pytest.raises(AgentError) as exc:
            configure(fixture, **change)
        assert exc.value.code == 'STORAGE_CONFIGURE_' + code
    fixture['state'].create_attempt(fixture['payload'], str(Path(fixture['config']['output_root']) / 'test-attempt'))
    with pytest.raises(AgentError) as exc:
        configure(fixture)
    assert exc.value.code == 'STORAGE_CONFIGURE_BUSY'


def test_preflight_checkout_and_profile_locks_prevent_a_storage_change(storage_fixture):
    fixture = storage_fixture
    for kind in ('background', 'profile', 'checkout'):
        if kind == 'background':
            with (Path(fixture['config']['state_dir']) / 'preflight-worker.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with pytest.raises(AgentError) as exc:
                    configure(fixture)
        elif kind == 'profile':
            with profile_configuration_lock(fixture['config']):
                with pytest.raises(AgentError) as exc:
                    configure(fixture)
        else:
            with checkout_lock(fixture['config'], str(fixture['repo'])):
                with pytest.raises(AgentError) as exc:
                    configure(fixture)
        assert exc.value.code == 'STORAGE_CONFIGURE_BUSY'


def test_failed_probe_or_persistence_keeps_original_settings(storage_fixture, monkeypatch):
    fixture = storage_fixture
    original = fixture['path'].read_bytes()
    live = copy.deepcopy(fixture['config'])
    with monkeypatch.context() as patch:
        patch.setattr(storage.tempfile, 'NamedTemporaryFile', lambda *a, **k: (_ for _ in ()).throw(PermissionError('fixture')))
        with pytest.raises(AgentError) as exc:
            configure(fixture)
        assert exc.value.code == 'STORAGE_CONFIGURE_UNWRITABLE'
    assert fixture['path'].read_bytes() == original and fixture['config'] == live
    monkeypatch.setattr(storage, '_persist', lambda *a, **k: (_ for _ in ()).throw(OSError('fixture disk full')))
    with pytest.raises(AgentError) as exc:
        configure(fixture)
    assert exc.value.code == 'STORAGE_CONFIGURE_FAILED'
    assert fixture['path'].read_bytes() == original and fixture['config'] == live


def test_cache_root_can_return_to_project_defaults_without_moving_old_cache(storage_fixture):
    fixture = storage_fixture
    first = configure(fixture)
    old_cache = Path(first['cache_root']) / 'keep-cache'
    old_cache.write_text('retained')
    request = {**fixture['request'], 'expected_output_root': first['output_root'],
               'expected_cache_root': first['cache_root'], 'cache_root': None}
    result = storage.configure_storage(fixture['config'], fixture['state'], request, operation_id='storage-2')
    assert result['cache_root'] is None and result['directories']['cache'] is None
    assert storage.cache_environment(fixture['config']) == {}
    assert old_cache.read_text() == 'retained'


def test_cache_environment_is_complete_and_does_not_change_agent_environment(storage_fixture):
    fixture = storage_fixture
    configure(fixture)
    old = dict(os.environ)
    environment = storage.cache_environment(fixture['config'])
    assert len(environment) == 11
    assert environment['HF_DATASETS_CACHE'] == str(Path(fixture['config']['cache_root']) / 'huggingface/datasets')
    assert environment['TMP'] == environment['TEMP'] == environment['TMPDIR']
    assert dict(os.environ) == old


def test_agent_dispatch_updates_monitor_but_keeps_old_attempt_output_roots(storage_fixture):
    fixture = storage_fixture
    old_root = fixture['config']['output_root']
    agent = Agent(fixture['config'], client=object())
    try:
        response = agent.handle_command({'id': 'storage-1', 'type': 'STORAGE_CONFIGURE', 'payload': fixture['request']})
        assert response['status'] == 'APPLIED'
        assert old_root in agent.artifact_monitor.output_roots
        old_attempt = Path(old_root) / 'historic-attempt'
        old_attempt.mkdir()
        monitor = ArtifactMonitor(fixture['state'], fixture['config']['output_root'], output_roots=[old_root],
                                  scanner=lambda *args: [])
        attempt = {'id': 'historic-attempt', 'output_dir': str(old_attempt),
                   'payload': {'recipe': 'test-recipe'}}
        assert monitor.report(attempt) == 0
    finally:
        agent.close()


@pytest.mark.parametrize('target', ['state', 'state-child', 'state-parent', 'config-parent'])
def test_private_agent_state_and_config_are_not_valid_storage_targets(storage_fixture, target):
    fixture = storage_fixture
    state = Path(fixture['config']['state_dir'])
    path = {'state': state, 'state-child': state / 'cache', 'state-parent': state.parent,
            'config-parent': fixture['path'].parent}[target]
    with pytest.raises(AgentError) as exc:
        configure(fixture, cache_root=str(path))
    assert exc.value.code == 'STORAGE_CONFIGURE_INVALID'


def test_actual_runner_child_uses_new_cache_paths_and_output_root(storage_fixture, monkeypatch):
    from vla_agent import runner
    from vla_agent.snapshots import capture_code
    fixture = storage_fixture
    configure(fixture)
    keys = list(storage.cache_environment(fixture['config']))
    marker = fixture['root'] / 'child-cache-environment.json'
    (fixture['repo'] / 'smoke_training.py').write_text(
        'import json,os\nfrom pathlib import Path\n'
        f'Path({str(marker)!r}).write_text(json.dumps({{key:os.environ.get(key) for key in {keys!r}}}))\n')
    fixture['payload']['code_expectation'] = capture_code(fixture['config']['profiles'][0], fixture['config']['roots'])
    output = Path(fixture['config']['output_root']) / fixture['payload']['attempt_id']
    fixture['state'].create_attempt(fixture['payload'], str(output))
    monkeypatch.setenv('HF_DATASETS_CACHE', '/old/inherited/cache')
    assert runner.execute(fixture['config'], fixture['payload']['attempt_id']) == 0
    assert fixture['state'].attempt(fixture['payload']['attempt_id'])['state'] == 'SUCCEEDED'
    assert output.is_dir()
    assert json.loads(marker.read_text()) == storage.cache_environment(fixture['config'])
    assert os.environ['HF_DATASETS_CACHE'] == '/old/inherited/cache'
