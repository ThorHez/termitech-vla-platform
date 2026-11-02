"""Real bounded metadata reads on local SSH-wire stand-ins; no datasets scanned."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from vla_control.dataset_statistics_transport import read_dataset_statistics


spec = importlib.util.spec_from_file_location('dataset_statistics_peer_helpers',
    Path(__file__).with_name('test_checkpoint_transfer_transport.py'))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
LocalRemote = helpers.LocalRemote


def fixtures(root):
    dataset = root / "dataset '; echo should-not-run #"
    (dataset / 'meta').mkdir(parents=True)
    info = {'fps': 20, 'total_frames': 1234, 'total_episodes': 12,
        'features': {'observation.state': {'dtype': 'float32', 'shape': [2, 7]},
            'observation.images.wrist': {'dtype': 'video', 'shape': [224, 224, 3]},
            'timestamp': {'dtype': 'float32', 'shape': []}},
        'private_token': 'must-not-leak'}
    norm = root / 'normalization' / 'norm_stats.json'
    norm.parent.mkdir()
    stats = {'norm_stats': {key: {'mean': [1, 2], 'std': [0.1, 0.2],
        'q01': [-1, -2], 'q99': [3, 4]} for key in ('state', 'actions')},
        'private_config': 'must-not-leak'}
    (dataset / 'meta/info.json').write_text(json.dumps(info))
    norm.write_text(json.dumps(stats))
    return dataset, norm, info, stats


def read(dataset, norm, roots):
    remote = LocalRemote()
    try:
        return read_dataset_statistics(remote, str(dataset), str(norm) if norm else None,
            [str(root) for root in roots])
    finally:
        for channel in remote.channels:
            channel.close()


def test_actual_metadata_counts_camera_fields_and_per_dimension_stats_are_whitelisted(tmp_path):
    dataset, norm, info, stats = fixtures(tmp_path)
    result = read(dataset, norm, [tmp_path])
    assert result['dataset']['status'] == result['normalization']['status'] == 'READY'
    assert result['dataset']['total_frames'] == 1234
    assert result['dataset']['total_episodes'] == 12
    assert result['dataset']['fps'] == 20 and result['dataset']['camera_count'] == 1
    assert result['dataset']['features'][0] == {'key': 'observation.state', 'dtype': 'float32', 'shape': [2, 7]}
    assert result['normalization']['features'][0] == {'key': 'state', 'dimension': 2, **stats['norm_stats']['state']}
    assert len(result['dataset']['source_sha256']) == len(result['normalization']['source_sha256']) == 64
    assert 'must-not-leak' not in json.dumps(result)
    assert result['dataset']['path'] == str(dataset / 'meta/info.json')
    assert result['normalization']['path'] == str(norm)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([dataset.name, norm.parent.name])


def test_norm_directory_registration_resolves_only_its_norm_stats_json(tmp_path):
    dataset, norm, _, _ = fixtures(tmp_path)
    result = read(dataset, norm.parent, [tmp_path])
    assert result['normalization']['status'] == 'READY'
    assert result['normalization']['path'] == str(norm)


def test_unbound_skips_norm_file_and_missing_norm_keeps_valid_dataset_summary(tmp_path):
    dataset, norm, _, _ = fixtures(tmp_path)
    assert read(dataset, None, [tmp_path])['normalization'] == {'status': 'UNBOUND', 'features': []}
    norm.unlink()
    result = read(dataset, norm, [tmp_path])
    assert result['dataset']['status'] == 'READY'
    assert result['normalization']['status'] == 'FAILED'
    assert result['normalization']['error']['code'] == 'DATASET_STATISTICS_NOT_FOUND'
    assert not result['normalization']['features']


def test_missing_info_keeps_existing_normalization_values_available(tmp_path):
    dataset, norm, _, _ = fixtures(tmp_path)
    (dataset / 'meta/info.json').unlink()
    result = read(dataset, norm, [tmp_path])
    assert result['dataset']['status'] == 'FAILED'
    assert result['normalization']['status'] == 'READY'


@pytest.mark.parametrize('target', ['info_file', 'info_parent', 'dataset', 'norm_file', 'norm_parent'])
def test_symbolic_links_are_rejected_in_every_file_and_directory_component(tmp_path, target):
    dataset, norm, _, _ = fixtures(tmp_path)
    if target == 'info_file':
        source = dataset / 'meta/info.json'
    elif target == 'info_parent':
        source = dataset / 'meta'
    elif target == 'dataset':
        source = dataset
    elif target == 'norm_file':
        source = norm
    else:
        source = norm.parent
    destination = source.with_name(source.name + '.real')
    source.rename(destination)
    source.symlink_to(destination)
    result = read(dataset, norm, [tmp_path])
    affected = 'dataset' if target.startswith('info') or target == 'dataset' else 'normalization'
    assert result[affected]['status'] == 'FAILED'
    assert result[affected]['error']['code'] == 'DATASET_STATISTICS_UNSAFE_FILE'


@pytest.mark.parametrize('target', ['info', 'norm'])
def test_oversized_json_is_not_loaded_and_does_not_hide_the_other_valid_file(tmp_path, target):
    dataset, norm, _, _ = fixtures(tmp_path)
    path = dataset / 'meta/info.json' if target == 'info' else norm
    with path.open('wb') as stream:
        stream.truncate((1024 * 1024 if target == 'info' else 8 * 1024 * 1024) + 1)
    result = read(dataset, norm, [tmp_path])
    affected = 'dataset' if target == 'info' else 'normalization'
    assert result[affected]['error']['code'] == 'DATASET_STATISTICS_SIZE_LIMIT'
    assert result['normalization' if target == 'info' else 'dataset']['status'] == 'READY'


@pytest.mark.parametrize('content', ['{"norm_stats":NaN}', '{"norm_stats":Infinity}',
    '{"norm_stats":{},"norm_stats":{}}', '[1,2]', 'not json', ''])
def test_invalid_or_ambiguous_json_never_reaches_public_statistics(tmp_path, content):
    dataset, norm, _, _ = fixtures(tmp_path)
    norm.write_text(content)
    result = read(dataset, norm, [tmp_path])
    assert result['normalization']['status'] == 'FAILED'
    assert result['normalization']['features'] == []
    assert result['dataset']['status'] == 'READY'


@pytest.mark.parametrize('change', ['mean_bool', 'negative_std', 'mismatch', 'too_many_dimensions',
    'reversed_quantiles', 'reversed_range', 'empty_feature', 'missing_std', 'nested'])
def test_invalid_norm_numeric_contracts_return_no_fabricated_values(tmp_path, change):
    dataset, norm, _, stats = fixtures(tmp_path)
    values = stats['norm_stats']['state']
    if change == 'mean_bool': values['mean'][0] = True
    elif change == 'negative_std': values['std'][0] = -1
    elif change == 'mismatch': values['q01'] = [-1]
    elif change == 'too_many_dimensions': values['mean'] = [0] * 4097
    elif change == 'reversed_quantiles': values['q01'][0] = 10
    elif change == 'reversed_range': values.update(min=[3, 4], max=[1, 2])
    elif change == 'empty_feature': stats['norm_stats']['state'] = {}
    elif change == 'missing_std': values.pop('std')
    else: values['mean'] = [[1], [2]]
    norm.write_text(json.dumps(stats))
    result = read(dataset, norm, [tmp_path])
    assert result['normalization']['status'] == 'FAILED'
    assert result['normalization']['error']['code'] == 'DATASET_STATISTICS_INVALID_NORM'
    assert result['normalization']['features'] == []


def test_norms_support_actual_raw_feature_keys_and_leave_missing_quantiles_absent(tmp_path):
    dataset, norm, _, _ = fixtures(tmp_path)
    norm.write_text(json.dumps({'norm_stats': {'observation.raw_position': {'mean': [1], 'std': [0]}}}))
    result = read(dataset, norm, [tmp_path])
    assert result['normalization']['status'] == 'READY'
    assert result['normalization']['features'] == [{'key': 'observation.raw_position', 'dimension': 1,
        'mean': [1], 'std': [0]}]


@pytest.mark.parametrize('change', ['negative_frames', 'boolean_episodes', 'invalid_fps', 'shape', 'feature_limit'])
def test_invalid_known_info_fields_fail_explicitly_without_losing_norm_stats(tmp_path, change):
    dataset, norm, info, _ = fixtures(tmp_path)
    if change == 'negative_frames': info['total_frames'] = -1
    elif change == 'boolean_episodes': info['total_episodes'] = True
    elif change == 'invalid_fps': info['fps'] = 0
    elif change == 'shape': info['features']['observation.state']['shape'] = [True]
    else: info['features'] = {f'f{i}': {'dtype': 'float32', 'shape': [1]} for i in range(257)}
    (dataset / 'meta/info.json').write_text(json.dumps(info))
    result = read(dataset, norm, [tmp_path])
    assert result['dataset']['status'] == 'FAILED'
    assert result['dataset']['error']['code'] == 'DATASET_STATISTICS_INVALID_METADATA'
    assert result['normalization']['status'] == 'READY'


def test_fifo_metadata_is_rejected_without_blocking(tmp_path):
    dataset, norm, _, _ = fixtures(tmp_path)
    norm.unlink()
    os.mkfifo(norm)
    result = read(dataset, norm, [tmp_path])
    assert result['normalization']['status'] == 'FAILED'
    assert result['normalization']['error']['code'] == 'DATASET_STATISTICS_UNSAFE_FILE'
