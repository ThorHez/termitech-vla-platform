"""Portable scan script over local stand-ins for the SSH wire; no real machines."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import pytest

from vla_control import dataset_scan_transport as module
from vla_control.dataset_scan_transport import TransferError, scan_dataset_roots


spec = importlib.util.spec_from_file_location('dataset_scan_peer_helpers',
    Path(__file__).with_name('test_checkpoint_transfer_transport.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
LocalRemote = helpers.LocalRemote


def make_dataset(path, **metadata):
    (path / 'meta').mkdir(parents=True)
    info = {'fps': 30, 'robot_type': 'ARX5', 'total_frames': 100, 'total_episodes': 5,
            'features': {'action': {'dtype': 'float32', 'shape': [7], 'names': ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'gripper']}},
            'private_token': 'never-return-this-secret', **metadata}
    (path / 'meta/info.json').write_text(json.dumps(info, ensure_ascii=False))
    return path


def scan(parent, roots=None, **options):
    remote = LocalRemote()
    try:
        return scan_dataset_roots(remote, str(parent), [str(parent.parent)] if roots is None else roots, **options)
    finally:
        for channel in remote.channels:
            channel.close()


def test_scan_discovers_sorted_roots_with_bounded_whitelisted_metadata(tmp_path):
    parent = tmp_path / "数据 ';echo BAD; #"
    second = make_dataset(parent / 'b')
    first = make_dataset(parent / 'a', task_description='Fold\nthe cloth.')
    (first / 'meta/tasks.jsonl').write_text('{"task":"unused", "private_key":"secret"}\n')
    (second / 'meta/tasks.jsonl').write_text('{"task":"Pick cloth", "private_key":"secret"}\n')
    result = scan(parent)
    assert [row['name'] for row in result['entries']] == ['a', 'b']
    assert result['path'] == str(parent) and result['truncated'] is False
    assert [row['relative_path'] for row in result['entries']] == ['a', 'b']
    assert result['entries'][0]['summary'] == {'fps': 30, 'robot_type': 'ARX5', 'total_frames': 100,
                                              'total_episodes': 5, 'task_description': 'Fold\nthe cloth.'}
    assert result['entries'][1]['summary']['task_description'] == 'Pick cloth'
    assert result['entries'][0]['feature_signature'] == result['entries'][1]['feature_signature']
    assert result['entries'][0]['info_sha256'] == hashlib.sha256((first / 'meta/info.json').read_bytes()).hexdigest()
    assert 'never-return-this-secret' not in json.dumps(result) and 'private_' not in json.dumps(result)


def test_parent_dataset_is_included_and_no_samples_or_nested_datasets_are_scanned(tmp_path):
    parent = make_dataset(tmp_path / 'dataset')
    (parent / 'data').mkdir()
    os.mkfifo(parent / 'data/payload.parquet')
    make_dataset(parent / 'data/nested')
    result = scan(parent, max_depth=4)
    assert len(result['entries']) == 1 and result['entries'][0]['relative_path'] == '.'
    assert result['scanned_directories'] == 1 and not result['truncated']


def test_recursion_and_explicit_depth_truncation_are_distinct(tmp_path):
    parent = tmp_path / 'parent'
    make_dataset(parent / 'direct')
    make_dataset(parent / 'group/nested')
    direct = scan(parent, recursive=False)
    assert [row['relative_path'] for row in direct['entries']] == ['direct']
    assert not direct['truncated']
    limited = scan(parent, recursive=True, max_depth=1)
    assert [row['relative_path'] for row in limited['entries']] == ['direct']
    assert limited['truncated'] and any('depth' in warning for warning in limited['warnings'])
    complete = scan(parent, max_depth=2)
    assert [row['relative_path'] for row in complete['entries']] == ['direct', 'group/nested']
    assert not complete['truncated']


def test_dataset_limit_is_not_silently_reported_as_complete(tmp_path):
    parent = tmp_path / 'parent'
    make_dataset(parent / 'a')
    make_dataset(parent / 'b')
    result = scan(parent, max_datasets=1)
    assert [row['name'] for row in result['entries']] == ['a']
    assert result['truncated'] and any('dataset limit' in warning for warning in result['warnings'])


def test_hidden_links_unsafe_names_are_skipped_without_following_them(tmp_path):
    parent = tmp_path / 'parent'
    make_dataset(parent / 'visible')
    make_dataset(parent / '.hidden')
    external = make_dataset(tmp_path / 'external')
    (parent / 'linked').symlink_to(external, target_is_directory=True)
    make_dataset(parent / 'bad\\name')
    result = scan(parent)
    assert [row['name'] for row in result['entries']] == ['visible']
    assert result['skipped_directories'] >= 3
    assert not result['truncated']
    with pytest.raises(TransferError) as error:
        scan(parent / 'linked', [str(tmp_path)])
    assert error.value.code == 'DATASET_SCAN_PATH_UNSAFE'


@pytest.mark.parametrize('location', ['meta', 'info'])
def test_symlinked_metadata_is_listed_as_error_never_read(tmp_path, location):
    parent = tmp_path / 'parent'
    root = parent / 'unsafe'
    secret = make_dataset(tmp_path / 'secret')
    if location == 'meta':
        root.mkdir(parents=True)
        (root / 'meta').symlink_to(secret / 'meta', target_is_directory=True)
    else:
        (root / 'meta').mkdir(parents=True)
        (root / 'meta/info.json').symlink_to(secret / 'meta/info.json')
    result = scan(parent)
    assert len(result['entries']) == 1
    assert result['entries'][0]['error']['code'] == 'DATASET_SCAN_UNSAFE_METADATA'
    assert result['entries'][0]['info_sha256'] is None
    assert 'never-return-this-secret' not in json.dumps(result)


def test_malformed_duplicate_keys_nonfinite_and_large_metadata_do_not_hide_other_candidates(tmp_path):
    parent = tmp_path / 'parent'
    for name, content in [('bad-json', '{"private_token":"secret"'), ('duplicates', '{"features":{},"features":{}}'),
                          ('nonfinite', '{"features":{"a":{"dtype":"float32","shape":[7]}},"fps":NaN}'),
                          ('oversized', 'x' * (1024 * 1024 + 1))]:
        path = parent / name / 'meta';path.mkdir(parents=True)
        (path / 'info.json').write_text(content)
    make_dataset(parent / 'valid')
    result = scan(parent)
    assert len(result['entries']) == 5
    assert sum(row['error'] is None for row in result['entries']) == 1
    assert result['entries'][-1]['name'] == 'valid'
    assert 'private_token' not in json.dumps(result) and 'secret' not in json.dumps(result)


def test_special_metadata_file_is_rejected_without_blocking(tmp_path):
    path = tmp_path / 'dataset/meta';path.mkdir(parents=True)
    os.mkfifo(path / 'info.json')
    started = time.monotonic()
    result = scan(path.parent)
    assert result['entries'][0]['error']['code'] == 'DATASET_SCAN_UNSAFE_METADATA'
    assert time.monotonic() - started < 3


@pytest.mark.parametrize('path', ['relative', '/a/../b', '/bad\\path', '/bad\npath', '/bad\x00path'])
def test_invalid_path_is_rejected_before_starting_a_remote_process(path):
    remote = LocalRemote()
    with pytest.raises(TransferError):
        scan_dataset_roots(remote, path, ['/'])
    assert remote.channels == []


def test_scope_boundary_checks_components_not_string_prefixes(tmp_path):
    remote = LocalRemote()
    with pytest.raises(TransferError) as error:
        scan_dataset_roots(remote, str(tmp_path / 'data-other'), [str(tmp_path / 'data')])
    assert error.value.code == 'DATASET_SCAN_PATH_OUTSIDE_ROOTS'
    assert remote.channels == []


def test_cancel_and_deadline_close_only_the_owned_remote_channel(tmp_path):
    parent = make_dataset(tmp_path / 'dataset')
    for options, expected in [({'cancelled': lambda: True}, 'DATASET_SCAN_CANCELLED'),
                               ({'timeout': .0001}, 'DATASET_SCAN_TIMEOUT')]:
        remote = LocalRemote()
        with pytest.raises(TransferError) as error:
            scan_dataset_roots(remote, str(parent), [str(tmp_path)], **options)
        assert error.value.code == expected
        assert all(channel.process.poll() is not None for channel in remote.channels)


@pytest.mark.parametrize('mutation', ['foreign-path', 'private-summary', 'bad-fps', 'bad-name', 'bad-error', 'missing-hash'])
def test_untrusted_remote_responses_are_revalidated(tmp_path, monkeypatch, mutation):
    parent = make_dataset(tmp_path / 'dataset')
    original = scan(parent)
    response = deepcopy(original)
    entry = response['entries'][0]
    if mutation == 'foreign-path':entry['path'] = str(tmp_path / 'other')
    elif mutation == 'private-summary':entry['summary']['private_token'] = 'secret'
    elif mutation == 'bad-fps':entry['summary']['fps'] = '30'
    elif mutation == 'bad-name':entry['name'] = 'bad\x00name'
    elif mutation == 'bad-error':entry['error'] = {'code': 'UNSAFE', 'message': 'secret'}
    else:entry['info_sha256'] = None
    monkeypatch.setattr(module, '_remote_json', lambda *args, **kwargs: response)
    with pytest.raises(TransferError) as error:
        scan_dataset_roots(LocalRemote(), str(parent), [str(tmp_path)])
    assert error.value.code == 'DATASET_SCAN_INVALID_RESULT'


def test_extreme_numeric_metadata_does_not_crash_discovery(tmp_path):
    parent = make_dataset(tmp_path / 'dataset', fps=10**400)
    result = scan(parent)
    assert result['entries'][0]['error'] is None
    assert 'fps' not in result['entries'][0]['summary']


def test_total_metadata_budget_returns_explicit_partial_result(tmp_path):
    parent = tmp_path / 'parent'
    for index in range(11):
        make_dataset(parent / f'dataset-{index:02}', unused_padding='x' * 900000)
    result = scan(parent)
    assert result['truncated']
    assert len(result['entries']) < 11
    assert result['entries'][-1]['error']['code'] == 'DATASET_SCAN_TOTAL_LIMIT'
    assert any('8 MiB' in message for message in result['warnings'])
    assert len(json.dumps(result)) < 20000


def test_response_budget_returns_bounded_candidates_instead_of_oversized_response(tmp_path):
    parent = tmp_path / 'parent'
    deep = parent
    for index in range(7):
        deep /= str(index) + 'x' * 70
    for index in range(200):
        make_dataset(deep / f'dataset-{index:03}', task_description='任' * 2000)
    result = scan(parent, max_depth=8)
    assert result['truncated'] and len(result['entries']) < 200
    assert any('response-size' in message for message in result['warnings'])
    assert len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode()) < module.MAX_SCAN_RESULT_BYTES


def test_feature_signature_describes_format_not_dataset_identity_or_task(tmp_path):
    parent = tmp_path / 'parent'
    make_dataset(parent / 'a', task_description='Task A', fps=20)
    make_dataset(parent / 'b', task_description='Task B', fps=60)
    changed = make_dataset(parent / 'c')
    info = json.loads((changed / 'meta/info.json').read_text())
    info['features']['action']['shape'] = [14]
    (changed / 'meta/info.json').write_text(json.dumps(info))
    entries = scan(parent)['entries']
    assert entries[0]['feature_signature'] == entries[1]['feature_signature']
    assert entries[1]['feature_signature'] != entries[2]['feature_signature']
