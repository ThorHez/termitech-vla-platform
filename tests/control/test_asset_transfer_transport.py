"""Real direct rsync copies with isolated SSH peers and asset filesystem guards."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from vla_control import asset_transfer_transport as module
from vla_control.asset_transfer_transport import TransferError, transfer_asset


spec = importlib.util.spec_from_file_location('asset_sync_peer_helpers',
    Path(__file__).with_name('test_checkpoint_transfer_transport.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
LocalRemote = helpers.LocalRemote

direct_spec = importlib.util.spec_from_file_location('asset_direct_peer_helpers',
    Path(__file__).with_name('test_direct_asset_transfer.py'))
direct_helpers = importlib.util.module_from_spec(direct_spec)
direct_spec.loader.exec_module(direct_helpers)


@pytest.fixture
def asset(tmp_path):
    if not shutil.which('rsync'):
        pytest.skip('Real asset transfer tests require rsync')
    source = tmp_path / "数据集';echo BAD; #"
    target = tmp_path / '目标目录'
    (source / 'data' / 'empty').mkdir(parents=True)
    (source / 'train_state').mkdir()
    (source / 'data' / 'episode.parquet').write_bytes(bytes(range(256)) * 4096)
    (source / 'train_state' / 'retain-all-data').write_bytes(b'ordinary registered asset content')
    (source / '任务.json').write_text('{"task":"fold cloth"}')
    target.mkdir()
    return tmp_path, source, target


def transfer(asset, **changes):
    root, source, target = asset
    source_peer, target_peer = direct_helpers.direct_local_pair(root / 'direct-peers')
    args = dict(source_remote=source_peer, target_remote=target_peer, source_path=str(source),
                target_parent=str(target), transfer_id='asset-123', source_roots=[str(root)],
                target_roots=[str(root)], target_name='fold cloth v1')
    args.update(changes)
    try:
        return transfer_asset(**args)
    finally:
        for remote in (args['source_remote'], args['target_remote']):
            for channel in remote.channels:
                channel.close()


def test_plain_dataset_copies_every_file_empty_directory_and_unicode_names(asset):
    events = []
    result = transfer(asset, progress=events.append)
    final = Path(result['target_path'])
    assert final == asset[2] / 'fold cloth v1'
    assert result['asset_path'] == str(final)
    assert (final / 'data' / 'empty').is_dir()
    for source in asset[1].rglob('*'):
        if source.is_file():
            assert (final / source.relative_to(asset[1])).read_bytes() == source.read_bytes()
    assert (final / 'train_state' / 'retain-all-data').is_file()
    assert not (final / 'termitech-policy.json').exists()
    assert result['files_total'] == 3
    assert result['bytes_total'] == sum(p.stat().st_size for p in asset[1].rglob('*') if p.is_file())
    for item in result['manifest']['files']:
        assert hashlib.sha256((final / item['path']).read_bytes()).hexdigest() == item['sha256']
    assert events[-1]['stage'] == 'SUCCEEDED' and events[-1]['percent'] == 100
    assert not (asset[2] / '.asset-123.partial').exists()


def test_single_norm_file_is_copied_and_asset_path_identifies_file(asset):
    source = asset[0] / 'norm_stats.json'
    source.write_text('{"norm_stats":{"state":{"mean":[0]}}}')
    result = transfer(asset, source_path=str(source), target_name='normalization')
    final = Path(result['target_path'])
    assert result['asset_path'] == str(final / source.name)
    assert (final / source.name).read_bytes() == source.read_bytes()
    assert result['files_total'] == 1
    assert [p.name for p in final.iterdir()] == [source.name]


def test_empty_registered_directory_is_preserved(asset):
    source = asset[0] / 'empty'
    (source / 'camera-0').mkdir(parents=True)
    result = transfer(asset, source_path=str(source))
    assert result['files_total'] == 0 and result['bytes_total'] == 0
    assert (Path(result['asset_path']) / 'camera-0').is_dir()


def test_successful_retry_revalidates_owned_published_data_without_overwriting(asset):
    first = transfer(asset)
    inode = (Path(first['target_path']) / 'data' / 'episode.parquet').stat().st_ino
    second = transfer(asset)
    assert second == first
    assert (Path(first['target_path']) / 'data' / 'episode.parquet').stat().st_ino == inode


def test_existing_user_destination_is_never_adopted_or_replaced(asset):
    final = asset[2] / 'fold cloth v1'
    final.mkdir()
    (final / 'keep').write_text('user data')
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_TARGET_EXISTS'
    assert (final / 'keep').read_text() == 'user data'
    assert not (asset[2] / '.asset-123.asset-transfer.json').exists()


def test_unowned_partial_and_receipt_symlinks_are_not_touched(asset):
    partial = asset[2] / '.asset-123.partial'
    partial.mkdir()
    (partial / 'keep').write_text('do not delete')
    with pytest.raises(TransferError):
        transfer(asset)
    assert (partial / 'keep').read_text() == 'do not delete'
    shutil.rmtree(partial)
    receipt = asset[2] / '.asset-123.asset-transfer.json'
    destination = asset[0] / 'receipt-source'
    destination.write_text('unrelated')
    receipt.symlink_to(destination)
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_PARTIAL_CONFLICT'
    assert destination.read_text() == 'unrelated'


def test_cancellation_closes_channels_and_retry_resumes_only_owned_partial(asset):
    cancelled = False

    def progress(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':
            cancelled = True

    with pytest.raises(TransferError) as exc:
        transfer(asset, progress=progress, cancelled=lambda: cancelled)
    assert exc.value.code == 'TRANSFER_CANCELLED'
    partial = asset[2] / '.asset-123.partial'
    assert partial.is_dir()
    assert not (asset[2] / 'fold cloth v1').exists()
    (partial / 'data').mkdir(exist_ok=True)
    content = (asset[1] / 'data' / 'episode.parquet').read_bytes()
    (partial / 'data' / 'episode.parquet').write_bytes(content[:10000])
    result = transfer(asset)
    assert (Path(result['target_path']) / 'data' / 'episode.parquet').read_bytes() == content


def test_checksum_mismatch_blocks_publish_then_retry_repairs_only_partial(asset):
    def corrupt(event):
        if event['stage'] == 'VERIFYING':
            (asset[2] / '.asset-123.partial' / 'data' / 'episode.parquet').write_bytes(b'bad copy')
    with pytest.raises(TransferError) as exc:
        transfer(asset, progress=corrupt)
    assert exc.value.code == 'TRANSFER_CHECKSUM_MISMATCH'
    assert not (asset[2] / 'fold cloth v1').exists()
    final = Path(transfer(asset)['target_path'])
    assert (final / 'data' / 'episode.parquet').read_bytes() == (asset[1] / 'data' / 'episode.parquet').read_bytes()


def test_changed_source_is_not_published_or_silently_adopted_on_retry(asset):
    def mutate(event):
        if event['stage'] == 'VERIFYING':
            (asset[1] / '任务.json').write_text('changed source')
    with pytest.raises(TransferError) as exc:
        transfer(asset, progress=mutate)
    assert exc.value.code == 'TRANSFER_SOURCE_CHANGED'
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_SOURCE_CHANGED'
    assert not (asset[2] / 'fold cloth v1').exists()


def test_changed_directory_layout_and_unrelated_partial_entries_are_rejected(asset):
    def mutate(event):
        if event['stage'] == 'VERIFYING':
            (asset[1] / 'another empty').mkdir()
    with pytest.raises(TransferError) as exc:
        transfer(asset, progress=mutate)
    assert exc.value.code == 'TRANSFER_SOURCE_CHANGED'
    (asset[1] / 'another empty').rmdir()
    partial = asset[2] / '.asset-123.partial'
    (partial / 'unrelated empty').mkdir()
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_PARTIAL_CONFLICT'
    assert (partial / 'unrelated empty').is_dir()


@pytest.mark.parametrize('where', ['file', 'directory'])
def test_source_links_are_not_followed(asset, where):
    outside = asset[0] / 'outside'
    outside.mkdir()
    (outside / 'secret').write_text('private')
    link = asset[1] / 'linked'
    link.symlink_to(outside if where == 'directory' else outside / 'secret')
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_UNSAFE_FILE'
    assert not (asset[2] / 'fold cloth v1').exists()


def checkpoint_alias(asset):
    checkout = asset[0] / 'registered-project'
    checkout.mkdir()
    (checkout / 'checkpoints').symlink_to(asset[1].parent, target_is_directory=True)
    return checkout / 'checkpoints' / asset[1].name


@pytest.mark.parametrize('excluded', [False, True])
def test_registered_source_ancestor_alias_is_resolved_and_canonical_identity_is_frozen(asset, excluded):
    alias = checkpoint_alias(asset)
    result = transfer(asset, source_path=str(alias), exclude_train_state=excluded)
    final = Path(result['target_path'])
    assert (final / 'data' / 'episode.parquet').read_bytes() == (asset[1] / 'data' / 'episode.parquet').read_bytes()
    assert (final / 'train_state').exists() is not excluded
    receipt_path = asset[2] / '.asset-123.asset-transfer.json'
    before = receipt_path.read_bytes()
    identity = json.loads(before)['source']
    assert identity['path'] == str(asset[1]) and identity['registered_path'] == str(alias)
    assert identity['root_identity']['device'] == asset[1].stat().st_dev
    assert identity['root_identity']['inode'] == asset[1].stat().st_ino
    assert Path(transfer(asset, source_path=str(alias), exclude_train_state=excluded)['target_path']) == final
    assert receipt_path.read_bytes() == before


def test_source_file_alias_with_different_basename_is_rejected_before_target_prepare(asset):
    source = asset[0] / 'norm_stats.json'
    source.write_bytes(b'normalization statistics')
    alias = asset[0] / 'registered-norm-alias'
    alias.symlink_to(source)
    with pytest.raises(TransferError) as error:
        transfer(asset, source_path=str(alias))
    assert error.value.code == 'TRANSFER_PATH_INVALID'
    assert 'canonical file path' in error.value.message
    assert list(asset[2].iterdir()) == []


def test_source_file_under_aliased_ancestor_preserves_registered_basename(asset):
    folder = asset[0] / 'norm-directory'
    folder.mkdir()
    source = folder / 'norm_stats.json'
    source.write_bytes(b'normalization statistics')
    alias = asset[0] / 'registered-norm-directory'
    alias.symlink_to(folder, target_is_directory=True)
    registered = alias / source.name
    result = transfer(asset, source_path=str(registered))
    assert Path(result['asset_path']).read_bytes() == source.read_bytes()
    assert Path(result['asset_path']).name == registered.name


@pytest.mark.parametrize('unauthorized', ['lexical', 'physical'])
def test_source_alias_requires_lexical_and_physical_managed_root_authorization(asset, unauthorized):
    allowed = asset[0] / 'allowed'
    allowed.mkdir()
    outside = asset[0] / 'outside-source'
    outside.mkdir()
    if unauthorized == 'physical':
        alias = allowed / 'alias'
        alias.symlink_to(asset[1], target_is_directory=True)
    else:
        source = allowed / 'checkpoint'
        source.mkdir()
        (source / 'weights').write_bytes(b'authorized content')
        alias = outside / 'alias'
        alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(TransferError) as error:
        transfer(asset, source_path=str(alias), source_roots=[str(allowed)])
    assert error.value.code == 'TRANSFER_PATH_OUTSIDE_ROOTS'
    assert list(asset[2].iterdir()) == []


@pytest.mark.parametrize('where', ['file', 'directory'])
def test_root_alias_does_not_authorize_internal_symlinks(asset, where):
    alias = checkpoint_alias(asset)
    outside = asset[0] / 'other-content'
    outside.mkdir()
    (outside / 'private').write_bytes(b'do not dereference')
    (asset[1] / 'internal-link').symlink_to(outside if where == 'directory' else outside / 'private')
    with pytest.raises(TransferError) as error:
        transfer(asset, source_path=str(alias))
    assert error.value.code == 'TRANSFER_UNSAFE_FILE'
    assert list(asset[2].iterdir()) == []


def test_retargeted_source_alias_is_rejected_before_publication(asset):
    alias = asset[0] / 'registered-root'
    alias.symlink_to(asset[1], target_is_directory=True)
    other = asset[0] / 'other-checkpoint'
    shutil.copytree(asset[1], other)
    def retarget(event):
        if event['stage'] == 'VERIFYING':
            alias.unlink()
            alias.symlink_to(other, target_is_directory=True)
    with pytest.raises(TransferError) as error:
        transfer(asset, source_path=str(alias), progress=retarget)
    assert error.value.code == 'TRANSFER_SOURCE_CHANGED'
    assert not (asset[2] / 'fold cloth v1').exists()
    assert (asset[2] / '.asset-123.partial').is_dir()


@pytest.mark.parametrize('change', ['alias-target', 'same-path-new-inode'])
def test_alias_resume_does_not_adopt_another_root_even_with_identical_bytes(asset, change):
    alias = asset[0] / 'registered-root'
    alias.symlink_to(asset[1], target_is_directory=True)
    cancelled = False
    def cancel(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':cancelled = True
    with pytest.raises(TransferError):
        transfer(asset, source_path=str(alias), progress=cancel, cancelled=lambda: cancelled)
    receipt_path = asset[2] / '.asset-123.asset-transfer.json'
    original = receipt_path.read_bytes()
    if change == 'alias-target':
        other = asset[0] / 'other-checkpoint'
        shutil.copytree(asset[1], other)
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
    else:
        retained = asset[0] / 'original-checkpoint'
        asset[1].rename(retained)
        shutil.copytree(retained, asset[1])
    with pytest.raises(TransferError) as error:
        transfer(asset, source_path=str(alias))
    assert error.value.code == 'TRANSFER_SOURCE_CHANGED'
    assert receipt_path.read_bytes() == original
    assert not (asset[2] / 'fold cloth v1').exists()


def test_source_alias_is_not_followed_in_target_modes_or_same_machine_overlap(asset):
    alias = checkpoint_alias(asset)
    target_alias = asset[0] / 'target-alias'
    target_alias.symlink_to(asset[2], target_is_directory=True)
    with pytest.raises(TransferError) as error:
        transfer(asset, source_path=str(alias), target_parent=str(target_alias))
    assert error.value.code == 'TRANSFER_PATH_INVALID'
    assert list(asset[2].iterdir()) == []
    with pytest.raises(TransferError) as error:
        transfer(asset, source_path=str(alias), target_parent=str(asset[1] / 'overlapping-child'))
    assert error.value.code == 'TRANSFER_PATH_OVERLAP'
    assert not (asset[1] / 'overlapping-child').exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', '.', '..', 'bad\nname', 'a\\b', ''])
def test_invalid_destination_names_are_rejected_before_transfer(asset, name):
    with pytest.raises(TransferError) as exc:
        transfer(asset, target_name=name)
    assert exc.value.code == 'TRANSFER_PATH_INVALID'


def test_managed_roots_and_canonical_paths_are_enforced(asset):
    with pytest.raises(TransferError) as exc:
        transfer(asset, target_roots=[str(asset[1])])
    assert exc.value.code == 'TRANSFER_PATH_OUTSIDE_ROOTS'
    with pytest.raises(TransferError) as exc:
        transfer(asset, source_path=str(asset[1] / '..' / asset[1].name))
    assert exc.value.code == 'TRANSFER_PATH_INVALID'


def test_same_host_aliases_cannot_copy_directory_into_itself(asset):
    parent = asset[1] / 'new' / 'nested'
    with pytest.raises(TransferError) as exc:
        transfer(asset, target_parent=str(parent))
    assert exc.value.code == 'TRANSFER_PATH_OVERLAP'
    assert not (asset[1] / 'new').exists(), 'Overlap is rejected before even creating target parents'


def test_distinct_machine_identity_does_not_reject_matching_path_strings(asset):
    source = module._asset_json(LocalRemote(), {'mode': 'source', 'path': str(asset[1]), 'roots': [str(asset[0])]}, lambda: False)
    # Only exercise target preparation: these local fixtures intentionally share
    # a filesystem, unlike the two machine identities represented by this test.
    prepared = module._asset_json(LocalRemote(), {
        'mode': 'prepare', 'parent': str(asset[1] / 'new remote directory'), 'roots': [str(asset[0])],
        'id': 'other-machine', 'name': 'copy', 'manifest': source['manifest'],
        'source_kind': source['kind'], 'source_name': source['name'], 'directories': source['directories'],
        'source_machine_id': 'a different boot identity', 'source_identity': {'path': str(asset[1])},
    }, lambda: False)
    assert Path(prepared['staging_path']).is_dir()


def test_space_and_manifest_limits_are_explicit_without_partial_publication(asset, monkeypatch):
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', module._ASSET_REMOTE_SCRIPT.replace('shutil.disk_usage(parent).free', '0'))
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_SPACE_INSUFFICIENT'
    assert not (asset[2] / 'fold cloth v1').exists()


def test_large_manifest_is_not_silently_truncated(asset, monkeypatch):
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', module._ASSET_REMOTE_SCRIPT.replace('MAX_FILES=100000', 'MAX_FILES=2'))
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_MANIFEST_LIMIT'
    assert not (asset[2] / 'fold cloth v1').exists()


def test_cancellation_before_hashing_never_creates_destination(asset):
    with pytest.raises(TransferError) as exc:
        transfer(asset, cancelled=lambda: True)
    assert exc.value.code == 'TRANSFER_CANCELLED'
    assert list(asset[2].iterdir()) == []


def test_cancellation_during_verification_does_not_publish_and_can_resume(asset):
    cancelled = False
    def cancel(event):
        nonlocal cancelled
        if event['stage'] == 'VERIFYING':
            cancelled = True
    with pytest.raises(TransferError) as exc:
        transfer(asset, progress=cancel, cancelled=lambda: cancelled)
    assert exc.value.code == 'TRANSFER_CANCELLED'
    assert not (asset[2] / 'fold cloth v1').exists()
    assert (asset[2] / '.asset-123.partial').is_dir()
    assert Path(transfer(asset)['target_path']).is_dir()


def test_corrupted_completed_destination_is_never_repaired_over_user_changes(asset):
    result = transfer(asset)
    changed = Path(result['asset_path']) / '任务.json'
    changed.write_text('new user contents')
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_TARGET_EXISTS'
    assert changed.read_text() == 'new user contents'


def test_special_files_are_rejected_without_opening_them(asset):
    import os
    os.mkfifo(asset[1] / 'fifo')
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_UNSAFE_FILE'


def test_target_parent_symlinks_and_reserved_destination_names_are_rejected(asset):
    link = asset[0] / 'linked target'
    link.symlink_to(asset[2])
    with pytest.raises(TransferError) as exc:
        transfer(asset, target_parent=str(link))
    assert exc.value.code == 'TRANSFER_PATH_INVALID'
    for name in ('.asset-123.partial', '.asset-123.asset-transfer.json'):
        with pytest.raises(TransferError) as exc:
            transfer(asset, target_name=name)
        assert exc.value.code == 'TRANSFER_PATH_INVALID'


def test_manifest_byte_budget_fails_explicitly_before_large_response_allocation(asset, monkeypatch):
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', module._ASSET_REMOTE_SCRIPT.replace('MAX_BYTES=64*1024*1024', 'MAX_BYTES=17000'))
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_MANIFEST_LIMIT'
    assert list(asset[2].iterdir()) == []


def test_interruption_after_atomic_publication_is_recoverable_without_an_empty_final(asset, monkeypatch):
    original = module._ASSET_REMOTE_SCRIPT
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', original.replace(
        '        rename_noreplace(staging,final)\n',
        '        rename_noreplace(staging,final)\n        os._exit(97)\n'))
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_REMOTE_FAILED'
    final = asset[2] / 'fold cloth v1'
    assert (final / 'data' / 'episode.parquet').is_file()
    assert not (asset[2] / '.asset-123.partial').exists()
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', original)
    assert Path(transfer(asset)['target_path']) == final


def test_publish_race_never_replaces_a_new_user_directory(asset, monkeypatch):
    original = module._ASSET_REMOTE_SCRIPT
    # The directory appears after the initial existence check, exactly where a
    # conventional os.rename would overwrite an empty directory owned by a user.
    script = original.replace('        rename_noreplace(staging,final)\n',
        '        final.mkdir()\n        rename_noreplace(staging,final)\n')
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', script)
    with pytest.raises(TransferError) as exc:
        transfer(asset)
    assert exc.value.code == 'TRANSFER_TARGET_EXISTS'
    assert (asset[2] / 'fold cloth v1').is_dir()
    assert (asset[2] / '.asset-123.partial').is_dir()


def test_cancellation_after_publish_receipt_does_not_report_success(asset, monkeypatch):
    original = module._asset_json
    cancelled = False
    def cancel_after_publish(remote, payload, callback):
        nonlocal cancelled
        value = original(remote, payload, callback)
        if payload['mode'] == 'publish':
            cancelled = True
        return value
    monkeypatch.setattr(module, '_asset_json', cancel_after_publish)
    with pytest.raises(TransferError) as exc:
        transfer(asset, cancelled=lambda: cancelled)
    assert exc.value.code == 'TRANSFER_CANCELLED'
    assert (asset[2] / 'fold cloth v1' / 'data' / 'episode.parquet').is_file()
    monkeypatch.setattr(module, '_asset_json', original)
    assert Path(transfer(asset)['target_path']).is_dir()


def test_model_exclusion_copies_only_root_train_state_directory_out_of_profile(asset):
    source = asset[1]
    (source / 'params').mkdir()
    (source / 'params' / 'weights').write_bytes(b'model parameters')
    (source / 'nested' / 'foo' / 'train_state').mkdir(parents=True)
    nested = source / 'nested' / 'foo' / 'train_state' / 'keep-optimizer-name'
    nested.write_bytes(b'nested user data remains part of the model asset')
    (source / 'train_state' / 'optimizer-state').write_bytes(b'optimizer' * 10000)

    def change_excluded_data(event):
        if event['stage'] == 'VERIFYING':
            (source / 'train_state' / 'optimizer-state').write_bytes(b'changed excluded state')

    result = transfer(asset, exclude_train_state=True, progress=change_excluded_data)
    final = Path(result['target_path'])
    assert not (final / 'train_state').exists()
    assert (final / 'params' / 'weights').read_bytes() == b'model parameters'
    assert (final / 'nested' / 'foo' / 'train_state' / nested.name).read_bytes() == nested.read_bytes()
    expected = [item for item in source.rglob('*') if item.is_file() and item.relative_to(source).parts[0] != 'train_state']
    assert result['files_total'] == len(expected)
    assert result['bytes_total'] == sum(item.stat().st_size for item in expected)
    assert all(not entry['path'].startswith('train_state/') for entry in result['manifest']['files'])
    receipt = json.loads((asset[2] / '.asset-123.asset-transfer.json').read_text())
    assert receipt['exclude_train_state'] is True
    assert 'train_state' not in receipt['directories']
    assert Path(transfer(asset, exclude_train_state=True)['target_path']) == final


def test_model_exclusion_never_hashes_or_descends_into_excluded_state(asset, monkeypatch):
    script = module._ASSET_REMOTE_SCRIPT.replace(
        '        if unsafe_name(display):',
        "        if display.startswith('train_state/'):fail('TRANSFER_EXCLUDED_READ','Excluded optimizer state was read')\n"
        '        if unsafe_name(display):')
    assert script != module._ASSET_REMOTE_SCRIPT
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', script)
    assert transfer(asset, exclude_train_state=True)['files_total'] == 2


def test_target_capacity_uses_excluded_manifest_instead_of_optimizer_state_size(asset, monkeypatch):
    (asset[1] / 'train_state' / 'optimizer-state').write_bytes(b'x' * 4 * 1024 * 1024)
    script = module._ASSET_REMOTE_SCRIPT.replace('if shutil.disk_usage(parent).free<', 'if 3*1024*1024<')
    assert script != module._ASSET_REMOTE_SCRIPT
    monkeypatch.setattr(module, '_ASSET_REMOTE_SCRIPT', script)
    result = transfer(asset, exclude_train_state=True)
    assert result['bytes_total'] < 2 * 1024 * 1024
    with pytest.raises(TransferError) as error:
        transfer(asset, transfer_id='full-model', target_name='full model', exclude_train_state=False)
    assert error.value.code == 'TRANSFER_SPACE_INSUFFICIENT'


def test_train_state_plain_file_is_kept_and_single_file_profile_is_a_noop(asset):
    shutil.rmtree(asset[1] / 'train_state')
    (asset[1] / 'train_state').write_bytes(b'this is a regular file, not optimizer state directory')
    result = transfer(asset, exclude_train_state=True)
    assert (Path(result['target_path']) / 'train_state').read_bytes() == (asset[1] / 'train_state').read_bytes()
    single = transfer(asset, source_path=str(asset[1] / 'train_state'), transfer_id='single-file',
                      target_name='single train state file', exclude_train_state=True)
    assert Path(single['asset_path']).read_bytes() == (asset[1] / 'train_state').read_bytes()


def test_source_asset_itself_named_train_state_is_not_wholly_excluded(asset):
    source = asset[1] / 'train_state'
    (source / 'params').write_bytes(b'parameters at asset root')
    (source / 'train_state').mkdir()
    (source / 'train_state' / 'excluded').write_bytes(b'only this inner root directory is excluded')
    result = transfer(asset, source_path=str(source), exclude_train_state=True)
    final = Path(result['target_path'])
    assert (final / 'params').read_bytes() == b'parameters at asset root'
    assert (final / 'retain-all-data').is_file()
    assert not (final / 'train_state').exists()


def test_empty_train_state_directory_is_not_counted_or_published(asset):
    shutil.rmtree(asset[1] / 'train_state')
    (asset[1] / 'train_state').mkdir()
    result = transfer(asset, exclude_train_state=True)
    assert result['files_total'] == 2
    assert not (Path(result['target_path']) / 'train_state').exists()


@pytest.mark.parametrize('nested', [False, True])
def test_train_state_symlinks_cannot_hide_behind_exclusion(asset, nested):
    outside = asset[0] / 'private-state'
    outside.mkdir()
    (outside / 'secret').write_bytes(b'not authorized by an exclusion rule')
    if nested:
        link = asset[1] / 'nested' / 'train_state'
        link.parent.mkdir()
    else:
        link = asset[1] / 'train_state'
        shutil.rmtree(link)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(TransferError) as error:
        transfer(asset, exclude_train_state=True)
    assert error.value.code == 'TRANSFER_UNSAFE_FILE'
    assert list(asset[2].iterdir()) == []


def test_legacy_false_receipt_shape_and_partial_retry_are_unchanged(asset):
    cancelled = False
    def cancel(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':cancelled = True
    with pytest.raises(TransferError):
        transfer(asset, progress=cancel, cancelled=lambda: cancelled)
    receipt_path = asset[2] / '.asset-123.asset-transfer.json'
    before = receipt_path.read_bytes()
    receipt = json.loads(before)
    assert set(receipt) == {'schema_version', 'transfer_id', 'target_name', 'source',
                            'source_kind', 'source_name', 'directories', 'manifest'}
    assert set(receipt['source']) == {'path', 'host', 'port', 'username'}
    result = transfer(asset, exclude_train_state=False)
    assert receipt_path.read_bytes() == before
    assert (Path(result['target_path']) / 'train_state' / 'retain-all-data').is_file()


def test_exclusion_cannot_adopt_a_partial_created_with_a_different_profile(asset):
    cancelled = False
    def cancel(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':cancelled = True
    with pytest.raises(TransferError):
        transfer(asset, progress=cancel, cancelled=lambda: cancelled)
    partial = asset[2] / '.asset-123.partial'
    (partial / 'train_state').mkdir()
    keep = partial / 'train_state' / 'keep-existing'
    keep.write_bytes(b'previous full-transfer partial files')
    with pytest.raises(TransferError) as error:
        transfer(asset, exclude_train_state=True)
    assert error.value.code == 'TRANSFER_PROFILE_CHANGED'
    assert 'new transfer directory' in error.value.message
    assert keep.read_bytes() == b'previous full-transfer partial files'
    assert 'exclude_train_state' not in json.loads((asset[2] / '.asset-123.asset-transfer.json').read_text())


def test_excluded_state_left_in_owned_partial_is_a_conflict_and_is_not_deleted(asset):
    cancelled = False
    def cancel(event):
        nonlocal cancelled
        if event['stage'] == 'TRANSFERRING':cancelled = True
    with pytest.raises(TransferError):
        transfer(asset, exclude_train_state=True, progress=cancel, cancelled=lambda: cancelled)
    partial = asset[2] / '.asset-123.partial'
    (partial / 'train_state').mkdir()
    keep = partial / 'train_state' / 'leftover'
    keep.write_bytes(b'never silently remove or publish old state')
    with pytest.raises(TransferError) as error:
        transfer(asset, exclude_train_state=True)
    assert error.value.code == 'TRANSFER_PARTIAL_CONFLICT'
    assert 'new transfer directory' in error.value.message
    assert keep.read_bytes() == b'never silently remove or publish old state'
    assert not (asset[2] / 'fold cloth v1').exists()


def test_excluded_state_introduced_after_copy_is_rejected_before_publish(asset):
    def mutate(event):
        if event['stage'] == 'VERIFYING':
            (asset[2] / '.asset-123.partial' / 'train_state').mkdir()
    with pytest.raises(TransferError) as error:
        transfer(asset, exclude_train_state=True, progress=mutate)
    assert error.value.code == 'TRANSFER_PARTIAL_CONFLICT'
    assert not (asset[2] / 'fold cloth v1').exists()


def test_excluded_state_introduced_after_inspect_is_rejected_at_publish(asset, monkeypatch):
    original = module._asset_json
    def mutate_before_publish(remote, payload, cancelled):
        if payload['mode'] == 'publish':
            (asset[2] / '.asset-123.partial' / 'train_state').mkdir()
        return original(remote, payload, cancelled)
    monkeypatch.setattr(module, '_asset_json', mutate_before_publish)
    with pytest.raises(TransferError) as error:
        transfer(asset, exclude_train_state=True)
    assert error.value.code == 'TRANSFER_PARTIAL_CONFLICT'
    assert not (asset[2] / 'fold cloth v1').exists()


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}])
def test_exclusion_profile_requires_a_strict_boolean_before_remote_work(asset, value):
    with pytest.raises(TransferError) as error:
        transfer(asset, exclude_train_state=value)
    assert error.value.code == 'TRANSFER_PROFILE_INVALID'
    assert list(asset[2].iterdir()) == []
