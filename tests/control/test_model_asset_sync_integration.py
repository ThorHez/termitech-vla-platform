"""MODEL filtering from authenticated API through real isolated direct rsync."""

import hashlib
import importlib.util
from pathlib import Path

from vla_control.asset_transfer_models import AssetTransfer
from vla_control.models import Asset, Node

from .test_asset_transfers import create, fixture


spec = importlib.util.spec_from_file_location('model_sync_direct_peer_helpers',
    Path(__file__).with_name('test_direct_asset_transfer.py'))
peers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peers)


def test_model_checkbox_policy_controls_real_direct_files_and_registered_metadata(fixture, tmp_path, monkeypatch):
    platform, manager, source_node, target_node, registered, connections = fixture
    model = tmp_path / 'source-model'
    (model / 'params').mkdir(parents=True)
    (model / 'train_state').mkdir()
    (model / 'nested/train_state').mkdir(parents=True)
    weights = b'unchanged model parameters\0' * 4096
    (model / 'params/weights.bin').write_bytes(weights)
    (model / 'train_state/optimizer.bin').write_bytes(b'optimizer state' * 8192)
    (model / 'nested/train_state/keep.bin').write_bytes(b'not the root state directory')
    destination = tmp_path / 'copies'
    destination.mkdir()
    with platform.app.state.db.write() as session:
        for node_id in (source_node['node_id'], target_node['node_id']):
            session.get(Node, node_id).roots = [str(tmp_path)]
        asset = session.get(Asset, registered['id'])
        asset.kind, asset.path = 'MODEL', str(model)
        asset.metadata_json = {'description': 'isolated model sync fixture'}
    source_peer, target_peer = peers.direct_local_pair(tmp_path / 'peers')
    source_peer.connect = target_peer.connect = lambda: None
    monkeypatch.setattr(manager.remote_manager, '_remote',
                        lambda connection: source_peer if connection.id == connections[0].id else target_peer)
    try:
        for excluded, name in ((True, 'weights-only'), (False, 'full-model')):
            response = create(fixture, target_directory=str(destination), target_name=name,
                              exclude_train_state=excluded)
            assert response.status_code == 202, response.text
            queued = response.json()
            assert queued['exclude_train_state'] is excluded
            manager._run(queued['id'])
            with platform.app.state.db.read() as session:
                transfer = session.get(AssetTransfer, queued['id'])
                assert transfer.state == 'SUCCEEDED', transfer.error
                copied = session.get(Asset, transfer.target_asset_id)
                assert copied.kind == 'MODEL'
                assert copied.metadata_json['asset_sync']['training_preflight_required'] is True
                assert bool(copied.metadata_json['asset_sync'].get('exclude_train_state')) is excluded
                paths = {item['path'] for item in transfer.manifest['files']}
                assert ('train_state/optimizer.bin' not in paths) is excluded
                assert 'nested/train_state/keep.bin' in paths
                if excluded:
                    assert copied.metadata_json['asset_sync']['excluded_root_directories'] == ['train_state']
                output = Path(copied.path)
            assert (output / 'params/weights.bin').read_bytes() == weights
            assert (output / 'nested/train_state/keep.bin').is_file()
            assert (not (output / 'train_state').exists()) is excluded
        assert hashlib.sha256((model / 'params/weights.bin').read_bytes()).digest() == hashlib.sha256(weights).digest()
        assert (model / 'train_state/optimizer.bin').is_file()
    finally:
        source_peer.close()
        target_peer.close()
