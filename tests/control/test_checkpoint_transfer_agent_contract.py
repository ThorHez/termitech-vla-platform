"""Real rsync receipts are consumed unchanged by controller and target Agent."""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import psutil
import pytest

from vla_adapters.portable_policy import export_portable_policy, portable_digest
from vla_agent.policy_deployment import _portable
from vla_agent.safety import AgentError
from vla_control.checkpoint_transfers import _manifest
from vla_control.checkpoint_transfer_transport import (
    TransferError, _RECEIVER_SCRIPT, _RELAY_SCRIPT, _python_command, transfer_checkpoint,
)


HELPER = Path(__file__).with_name('test_checkpoint_transfer_transport.py')
loader = importlib.util.spec_from_file_location('rsync_contract_test_helper', HELPER)
helper = importlib.util.module_from_spec(loader)
loader.loader.exec_module(helper)


@dataclasses.dataclass(frozen=True)
class Config:
    model: object
    data: object
    policy_metadata: dict
    assets_dirs: tuple = ()


def portable_configuration(root, monkeypatch):
    """Export real safe JSON while replacing only the optional GPU model module."""
    classes = {}
    for name, fields in (
        ('openpi.models.pi0_config', [('action_dim', int, 32), ('action_horizon', int, 50)]),
        ('openpi.transforms', [('inputs', tuple, ()), ('outputs', tuple, ())]),
    ):
        module = ModuleType(name)
        source = root / 'src' / (name.replace('.', '/') + '.py')
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# portable integration fixture: ' + name + '\n')
        module.__file__ = str(source)
        cls = dataclasses.make_dataclass('Pi0Config' if name.endswith('pi0_config') else 'Group', fields, frozen=True)
        cls.__module__ = name
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, name, module)
        classes[cls.__name__] = cls
    data = SimpleNamespace(asset_id='.', data_transforms=classes['Group'](), model_transforms=classes['Group'](),
        use_quantile_norm=True, normalize_masks=None, robot_type=None)
    factory = SimpleNamespace(create=lambda *_: data)
    exported = export_portable_policy(Config(classes['Pi0Config'](), factory, {'task': '折叠衣服'}), root)
    return {'config': exported, 'digest': portable_digest(exported)}


def test_real_transfer_manifest_controller_and_target_agent_agree(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    (source / 'params' / 'd').mkdir(parents=True)
    (source / 'params' / 'd' / 'weights').write_bytes(bytes(range(256)) * 1000)
    (source / 'assets').mkdir()
    (source / 'assets' / 'norm_stats.json').write_text('{"state":{}}')
    (source / '训练信息.json').write_text('{"训练":"折叠衣服"}')
    package = portable_configuration(tmp_path / 'checkout', monkeypatch)
    contents = json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result = transfer_checkpoint(helper.LocalRemote(), helper.LocalRemote(), str(source), str(tmp_path / 'target'),
        'verified-transfer', [str(tmp_path)], [str(tmp_path)], extra_files={'termitech-policy.json': contents})
    assert _manifest(result['manifest'], package) == result['manifest']
    assert result['manifest_sha256'] == portable_digest(result['manifest'])
    target = Path(result['target_path'])
    payload = {'portable_policy': {**package, 'manifest': result['manifest'],
        'checkpoint_manifest_digest': result['manifest_sha256'], 'transfer_id': 'verified-transfer'}}
    accepted = _portable({'roots': [str(tmp_path)]}, target, payload)
    assert accepted['portable_policy'] == package
    assert accepted['checkpoint_manifest_digest'] == result['manifest_sha256']
    assert accepted['transfer_id'] == 'verified-transfer'
    # Same size defeats size-only validation; target must recompute SHA-256.
    weights = target / 'params' / 'd' / 'weights'
    original = weights.read_bytes()
    weights.write_bytes(b'evil' + original[4:])
    with pytest.raises(AgentError) as failure:
        _portable({'roots': [str(tmp_path)]}, target, payload)
    assert failure.value.code == 'POLICY_TRANSFER_INTEGRITY_INVALID'
    assert 'checksum' in str(failure.value)


@pytest.mark.parametrize('termination', [signal.SIGINT, signal.SIGHUP])
def test_receiver_signal_reaps_rsync_and_private_rsh_processes(tmp_path, termination):
    import base64
    target = tmp_path / 'partial'
    target.mkdir()
    command = _python_command(_RECEIVER_SCRIPT, {'path': str(target), 'relay': base64.b64encode(_RELAY_SCRIPT.encode()).decode()})
    wrapper = subprocess.Popen(['/bin/sh', '-c', command], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    owned = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            owned = psutil.Process(wrapper.pid).children(recursive=True)
            # Wait for both receiver and its real remote-shell helper, proving
            # that terminating the wrapper must clean the separate child group.
            if len(owned) >= 2:
                break
            time.sleep(.02)
        assert len(owned) >= 2
        os.kill(wrapper.pid, termination)
        wrapper.wait(timeout=7)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(child.is_running() and child.status() != psutil.STATUS_ZOMBIE for child in owned):
            time.sleep(.02)
        assert all(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE for child in owned)
    finally:
        if wrapper.poll() is None:
            os.killpg(wrapper.pid, signal.SIGTERM)
            wrapper.wait(timeout=7)
        wrapper.stdin.close()
        wrapper.stdout.close()
        wrapper.stderr.close()


def test_cancel_closes_both_ssh_rsync_channels_without_orphans(tmp_path):
    root, source, target = helper.checkpoint.__wrapped__(tmp_path)
    source_remote, target_remote = helper.LocalRemote(), helper.LocalRemote()
    cancelled, owned = False, []
    def progress(event):
        nonlocal cancelled, owned
        if event['stage'] != 'TRANSFERRING' or cancelled:
            return
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            owned = []
            for remote in (source_remote, target_remote):
                process = psutil.Process(remote.channels[-1].process.pid)
                owned.extend([process, *process.children(recursive=True)])
            if len(owned) >= 4:
                break
            time.sleep(.02)
        assert len(owned) >= 4  # sender, receiver wrapper, rsync client and rsh
        cancelled = True
    with pytest.raises(TransferError) as failure:
        transfer_checkpoint(source_remote, target_remote, str(source), str(target), 'cancel-owned',
            [str(root)], [str(root)], progress=progress, cancelled=lambda:cancelled)
    assert failure.value.code == 'TRANSFER_CANCELLED'
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(child.is_running() and child.status() != psutil.STATUS_ZOMBIE for child in owned):
        time.sleep(.02)
    assert all(not child.is_running() or child.status() == psutil.STATUS_ZOMBIE for child in owned)
    assert not (target / 'cancel-owned').exists()
    assert (target / '.cancel-owned.partial').is_dir()
