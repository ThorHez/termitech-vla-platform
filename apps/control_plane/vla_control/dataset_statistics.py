"""Read registered dataset statistics without scheduling work on a training Agent."""

from copy import deepcopy
from pathlib import PurePosixPath
import threading

from fastapi.responses import JSONResponse

from .checkpoint_transfers import _connection
from .common import APIError, now, offset_time
from .dataset_statistics_transport import (normalize_info, normalize_norm, public_error,
                                           read_dataset_statistics)
from .models import Asset, Node
from .dataset_scan_transport import _path, _under


def _error(code, message):
    return {'code': code, 'message': message}


def _failed(path=None, error=None, *, status='FAILED', asset_id=None):
    return {'status': status, 'features': [],
            **({'path': path} if path else {}),
            **({'asset_id': asset_id} if asset_id else {}),
            **({'error': error} if error else {})}


def _clean_section(value, path, kind):
    """Expose a final whitelist even when the transport implementation changes."""
    error = public_error('DATASET_STATISTICS_INVALID_RESULT')
    if not isinstance(value, dict):
        return _failed(path, error)
    allowed_paths = {path, str(PurePosixPath(path) / ('meta/info.json' if kind == 'dataset' else 'norm_stats.json'))}
    actual_path = value.get('path')
    if not isinstance(actual_path, str) or actual_path not in allowed_paths:
        return _failed(path, error)
    if value.get('status') != 'READY':
        code = value.get('error', {}).get('code') if isinstance(value.get('error'), dict) else None
        return _failed(actual_path, public_error(code))
    try:
        features = value.get('features')
        if not isinstance(features, list) or not 1 <= len(features) <= 256:
            raise ValueError('invalid features')
        fields = {}
        for feature in features:
            if not isinstance(feature, dict) or not isinstance(feature.get('key'), str) or feature['key'] in fields:
                raise ValueError('invalid feature entry')
            fields[feature['key']] = {key: deepcopy(item) for key, item in feature.items() if key != 'key'}
        if kind == 'dataset':
            clean = normalize_info({**{key: value[key] for key in ('total_frames', 'total_episodes', 'fps') if key in value},
                                    'features': fields})
        else:
            clean = normalize_norm({'norm_stats': fields})
        sha = value.get('source_sha256')
        if sha is not None and (not isinstance(sha, str) or len(sha) != 64 or any(char not in '0123456789abcdef' for char in sha)):
            raise ValueError('invalid source digest')
        return {'status': 'READY', 'path': actual_path, **clean,
                **({'source_sha256': sha} if sha else {})}
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        return _failed(path, error)


class DatasetStatisticsManager:
    def __init__(self, db, settings, remote_manager, *, reader=None):
        self.db, self.settings, self.remote_manager = db, settings, remote_manager
        self.reader = reader or read_dataset_statistics
        self._slots = threading.BoundedSemaphore(4)

    def _snapshot(self, session, asset_id):
        dataset = session.get(Asset, asset_id)
        if dataset is None:
            raise APIError(404, 'NOT_FOUND', '数据集资产不存在。')
        if dataset.kind != 'DATASET':
            raise APIError(422, 'DATASET_STATISTICS_KIND_INVALID', '只有数据集资产可以读取数据统计。')
        node = session.get(Node, dataset.node_id)
        norm = session.get(Asset, dataset.norm_asset_id) if dataset.norm_asset_id else None
        return {'asset_id': dataset.id, 'node_id': dataset.node_id, 'dataset_path': dataset.path,
                'dataset_state': dataset.state, 'norm_asset_id': dataset.norm_asset_id,
                'roots': deepcopy(node.roots) if node else [],
                'node_status': node.status if node else None,
                'last_seen': node.last_seen if node else None,
                'norm': {key: getattr(norm, key) for key in ('id', 'kind', 'node_id', 'path', 'state')} if norm else None}

    @staticmethod
    def _identity(snapshot):
        return {key: value for key, value in snapshot.items() if key != 'last_seen'}

    def get(self, asset_id):
        with self.db.read() as session:
            snapshot = self._snapshot(session, asset_id)
        dataset_path, norm = snapshot['dataset_path'], snapshot['norm']
        result = {'asset_id': asset_id, 'node_id': snapshot['node_id'], 'status': 'FAILED', 'checked_at': now(),
                  'dataset': _failed(dataset_path), 'normalization': _failed(status='UNBOUND')}
        norm_id = snapshot['norm_asset_id']
        if norm_id:
            result['normalization'] = _failed(norm['path'] if norm else None, asset_id=norm_id)
        norm_path, norm_override = None, None
        if norm_id:
            if norm is None or norm['state'] in {'ARCHIVED', 'UNAVAILABLE'}:
                norm_override = _failed(norm['path'] if norm else None,
                    _error('DATASET_STATISTICS_NORM_UNAVAILABLE', '已绑定的归一化资产不存在、已归档或不可用。'),
                    status='UNAVAILABLE', asset_id=norm_id)
            elif norm['kind'] != 'NORM' or norm['node_id'] != snapshot['node_id']:
                norm_override = _failed(error=_error('DATASET_STATISTICS_NORM_BINDING_INVALID', '归一化绑定必须指向同一机器上的 NORM 资产。'), asset_id=norm_id)
            else:
                norm_path = norm['path']
        if norm_override:
            result['normalization'] = norm_override
        if (snapshot['node_status'] != 'ONLINE' or not isinstance(snapshot['last_seen'], str)
                or snapshot['last_seen'] < offset_time(-self.settings.heartbeat_timeout_seconds)):
            error = _error('DATASET_STATISTICS_OFFLINE', '机器离线或心跳已过期，暂时无法读取磁盘统计。')
            result.update(status='OFFLINE', error=error, dataset=_failed(dataset_path, error, status='OFFLINE'))
            if norm_path:
                result['normalization'] = _failed(norm_path, error, status='OFFLINE', asset_id=norm_id)
            return result
        try:
            roots = [_path(root) for root in snapshot['roots']]
            canonical_dataset = _path(dataset_path)
            if not roots or not any(_under(canonical_dataset, root) for root in roots):
                raise ValueError('dataset outside roots')
        except Exception:
            error = public_error('DATASET_STATISTICS_PATH_OUTSIDE_ROOTS')
            result.update(error=error, dataset=_failed(dataset_path, error))
            if norm_id and not norm_override:
                result['normalization'] = _failed(norm_path, error, asset_id=norm_id)
            return result
        if norm_path:
            try:
                canonical_norm = _path(norm_path)
                if not any(_under(canonical_norm, root) for root in roots):
                    raise ValueError('norm outside roots')
                norm_path = canonical_norm
            except Exception:
                norm_override = _failed(norm_path, public_error('DATASET_STATISTICS_PATH_OUTSIDE_ROOTS'), asset_id=norm_id)
                result['normalization'] = norm_override
                norm_path = None
        if not self._slots.acquire(blocking=False):
            error = _error('DATASET_STATISTICS_BUSY', '统计读取请求较多，请稍后重试。')
            result.update(error=error, dataset=_failed(dataset_path, error))
            if norm_id and not norm_override:
                result['normalization'] = _failed(norm_path, error, asset_id=norm_id)
            return result
        remote, timer = None, None
        timed_out = threading.Event()
        try:
            with self.db.read() as session:
                connection = _connection(session, snapshot['node_id'])
                remote = self.remote_manager._remote(connection)
            def expire():
                timed_out.set()
                try:
                    remote.close()
                except Exception:
                    pass
            timer = threading.Timer(12, expire)
            timer.daemon = True
            timer.start()
            remote.connect()
            value = self.reader(remote, canonical_dataset, norm_path, roots, timeout=8)
            if timed_out.is_set():
                raise TimeoutError()
            if not isinstance(value, dict):
                raise ValueError('invalid statistics result')
            dataset_value = _clean_section(value.get('dataset'), canonical_dataset, 'dataset')
            normalization = norm_override or (
                {**_clean_section(value.get('normalization'), norm_path, 'normalization'), 'asset_id': norm_id}
                if norm_path else _failed(status='UNBOUND'))
            with self.db.read() as session:
                current = self._snapshot(session, asset_id)
            if self._identity(current) != self._identity(snapshot):
                error = _error('DATASET_STATISTICS_BINDING_CHANGED', '资产绑定、路径或机器配置已变化，请重新读取。')
                result.update(error=error, dataset=_failed(dataset_path, error),
                    normalization=_failed(error=error, asset_id=norm_id))
                return result
            result.update(dataset=dataset_value, normalization=normalization, checked_at=now())
            if dataset_value['status'] == 'READY' and normalization['status'] in {'READY', 'UNBOUND'}:
                result['status'] = 'READY'
            elif 'READY' in {dataset_value['status'], normalization['status']}:
                result['status'] = 'PARTIAL'
            return result
        except Exception as exc:
            code = getattr(exc, 'code', None)
            if isinstance(exc, TimeoutError) or code in {'TRANSFER_TIMEOUT', 'SSH_COMMAND_TIMEOUT'} or timed_out.is_set():
                code = 'DATASET_STATISTICS_TIMEOUT'
            error = public_error(code)
            result.update(error=error, dataset=_failed(dataset_path, error))
            if norm_path:
                result['normalization'] = _failed(norm_path, error, asset_id=norm_id)
            return result
        finally:
            if timer:
                timer.cancel()
            if remote:
                try:
                    remote.close()
                except Exception:
                    pass
            self._slots.release()


def register_routes(app, admin, manager):
    @app.get('/api/v1/assets/{asset_id}/statistics', dependencies=admin)
    def get(asset_id: str):
        return JSONResponse(manager.get(asset_id), headers={'Cache-Control': 'no-store'})
