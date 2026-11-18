"""Real rsync receipts are consumed unchanged by controller and target Agent."""
import dataclasses
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters.portable_policy import export_portable_policy, portable_digest
from vla_agent.policy_deployment import _portable
from vla_agent.safety import AgentError
from vla_control.checkpoint_transfer_transport import TransferError, transfer_checkpoint
from vla_control.checkpoint_transfers import _manifest

from .test_direct_asset_transfer import peers


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


def test_real_transfer_manifest_controller_and_target_agent_agree(tmp_path, monkeypatch, peers):
    source = tmp_path / 'source'
    (source / 'params' / 'd').mkdir(parents=True)
    (source / 'params' / 'd' / 'weights').write_bytes(bytes(range(256)) * 1000)
    (source / 'assets').mkdir()
    (source / 'assets' / 'norm_stats.json').write_text('{"state":{}}')
    (source / '训练信息.json').write_text('{"训练":"折叠衣服"}')
    package = portable_configuration(tmp_path / 'checkout', monkeypatch)
    contents = json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result = transfer_checkpoint(*peers, str(source), str(tmp_path / 'target'),
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


def test_cancel_stops_direct_rsync_and_retains_partial_checkpoint(tmp_path, peers):
    source, target = tmp_path / 'source', tmp_path / 'target'
    (source / 'params').mkdir(parents=True)
    (source / 'params' / 'weights').write_bytes(b'weights')
    (source / 'assets').mkdir()
    (source / 'assets' / 'norm_stats.json').write_text('{"state":{}}')
    source_remote, target_remote = peers
    connection_attempt = tmp_path / 'connection-attempt'
    target_remote.environment['DIRECT_FAKE_SSH_COUNT'] = str(connection_attempt)
    target_remote.environment['DIRECT_FAKE_SSH_DELAY'] = '30'

    with pytest.raises(TransferError) as failure:
        transfer_checkpoint(source_remote, target_remote, str(source), str(target), 'cancel-owned',
            [str(tmp_path)], [str(tmp_path)], cancelled=connection_attempt.is_file)
    assert failure.value.code == 'TRANSFER_CANCELLED'
    assert all(channel.exit_status_ready() for remote in peers for channel in remote.channels)
    assert (source_remote.home / '.ssh' / 'authorized_keys').read_bytes() == b''
    assert not (target / 'cancel-owned').exists()
    assert (target / '.cancel-owned.partial').is_dir()
