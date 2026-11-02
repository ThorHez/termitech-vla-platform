"""Durable registered-asset copies using the existing pinned machine SSH access."""

from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import PurePosixPath
import re
import threading

from fastapi import Query
from fastapi.responses import Response
from pydantic import Field, field_validator
from sqlalchemy import or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import defer
from vla_protocol import canonical_digest

from .asset_transfer_models import (AssetTransfer, ASSET_TRANSFER_ACTIVE,
    ASSET_TRANSFER_RETRYABLE, ASSET_TRANSFER_TERMINAL, ASSET_TRANSFER_WORKING)
from .checkpoint_transfers import _connection, _online, _verify_saved_ssh_credentials
from .common import APIError, now, require_path, uid
from .dataset_validations import INTERNAL_KEY as DATASET_VALIDATION_KEY, queue_validation
from .models import Asset
from .remote_models import RemoteConnection
from .schemas import InputModel
from .ssh_remote import RemoteError
from .storage_configurations import require_not_pending


LOG = logging.getLogger(__name__)
PUBLIC_FIELDS = ('id', 'name', 'source_asset_id', 'source_node_id', 'source_path', 'source_kind',
    'target_node_id', 'target_directory', 'target_name', 'target_path', 'asset_path', 'target_asset_id',
    'include_norm_stats', 'norm_sync', 'exclude_train_state',
    'state', 'attempt', 'bytes_total', 'bytes_transferred', 'files_total', 'files_transferred',
    'speed_bytes_per_second', 'progress_pct', 'logs', 'error', 'manifest_sha256',
    'created_at', 'updated_at', 'finished_at')
SEMANTIC_METADATA = frozenset({'dataset_profile', 'description', 'task_description', 'format',
    'robot_type', 'robot_model', 'arm_mode', 'action_representation', 'fps', 'tags', 'notes'})


def _failure_details(error):
    """Keep explicit contracts and make empty transport exceptions actionable."""
    fallback = '数据同步失败，已传输的部分文件保留；可手动重试，若仍失败请检查同步日志。'
    code = getattr(error, 'code', None)
    if code:
        message = str(getattr(error, 'message', str(error)) or '').strip() or fallback
        return str(code)[:128], message
    if isinstance(error, TimeoutError):
        return ('TRANSFER_CONNECTION_TIMEOUT',
            '网络或 SSH 数据通道暂未响应，同步已停止；已传输的部分文件保留，可手动重试继续。')
    if isinstance(error, (ConnectionError, EOFError)):
        return ('TRANSFER_CONNECTION_CLOSED',
            '网络或 SSH 数据通道已断开；已传输的部分文件保留，可检查连接后手动重试继续。')
    return 'ASSET_TRANSFER_FAILED', str(error).strip() or fallback


def component(value):
    if (not value or value in {'.', '..'} or '/' in value or '\\' in value
            or len(value) > 160 or len(value.encode('utf-8')) > 255
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
        raise ValueError('目标目录名必须是单个非空名称，不能包含斜杠、控制字符或超过文件名长度限制')
    return value


class AssetTransferInput(InputModel):
    source_asset_id: str = Field(min_length=1, max_length=36)
    target_node_id: str = Field(min_length=1, max_length=36)
    target_directory: str = Field(min_length=1, max_length=4096)
    target_name: str | None = Field(default=None, min_length=1, max_length=160)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)
    include_norm_stats: bool = Field(default=False, strict=True)
    exclude_train_state: bool = Field(default=False, strict=True)
    expected_norm_asset_id: str | None = Field(default=None, min_length=1, max_length=36)

    @field_validator('target_name')
    @classmethod
    def target_component(cls, value):
        if value is None:
            return None
        if any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
            raise ValueError('目标目录名不能包含控制字符')
        return component(value.strip())

    @field_validator('name', 'idempotency_key')
    @classmethod
    def nonempty_text(cls, value):
        if value is not None and (not value.strip() or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)):
            raise ValueError('名称和幂等标识不能是空白或包含控制字符')
        return value.strip() if value is not None else value


def asset_transfer_dict(row):
    value = {key: deepcopy(getattr(row, key)) for key in PUBLIC_FIELDS}
    errors = [value['error']]
    if value['norm_sync']:
        errors.append(value['norm_sync'].get('error'))
    for error in errors:
        if isinstance(error, dict) and error.get('code') and not str(error.get('message') or '').strip():
            error['message'] = '本次同步未记录具体错误原因；已传文件保留，可手动重试继续。'
    if value['files_transferred'] == -1:
        value['files_transferred'] = None
    if value['norm_sync'] and value['norm_sync'].get('files_transferred') == -1:
        value['norm_sync']['files_transferred'] = None
    value['byte_verified'] = row.state == 'SUCCEEDED'
    value['training_preflight_required'] = True
    return value


def active_asset_transfers(session, node_id=None, asset_id=None, *, lock=False):
    query = select(AssetTransfer).where(AssetTransfer.deleted_at.is_(None), AssetTransfer.state.in_(ASSET_TRANSFER_ACTIVE))
    if node_id is not None:
        query = query.where(or_(AssetTransfer.source_node_id == node_id, AssetTransfer.target_node_id == node_id))
    if asset_id is not None:
        query = query.where(or_(AssetTransfer.source_asset_id == asset_id,
            AssetTransfer.norm_sync['source_asset_id'].as_string() == asset_id))
    if lock:
        query = query.order_by(AssetTransfer.id).with_for_update().execution_options(populate_existing=True)
    return list(session.scalars(query))


def _admission_lock(session):
    # All create/retry entrypoints claim this before Asset/Node row locks.
    # Workers can finish independently; only admission to the shared capacity
    # and overlapping target namespace needs serialization across processes.
    if session.get_bind().dialect.name == 'postgresql':
        key = int.from_bytes(hashlib.sha256(b'asset-transfer-admission').digest()[:8], 'big', signed=True)
        session.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': key})


def _path(value, roots, *, allow_root=False):
    if any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
        raise APIError(422, 'ASSET_TRANSFER_PATH_INVALID', '同步路径不能包含控制字符')
    require_path(value, roots)
    path = str(PurePosixPath(value))
    if path == '/' and not allow_root:
        raise APIError(422, 'ASSET_TRANSFER_PATH_INVALID', '请选择独立的数据目录')
    return path


def _identity(connection):
    return {key: getattr(connection, key) for key in ('id', 'host', 'port', 'username', 'fingerprint')}


def _bindings(session, body, settings):
    asset = session.scalar(select(Asset).where(Asset.id == body.source_asset_id).with_for_update())
    if asset is None:
        raise APIError(404, 'NOT_FOUND', '来源资产不存在')
    if asset.kind not in {'DATASET', 'NORM', 'MODEL'} or asset.state in {'ARCHIVED', 'UNAVAILABLE'}:
        raise APIError(409, 'ASSET_TRANSFER_SOURCE_UNAVAILABLE', '来源资产已归档、不可用或不支持同步')
    if body.exclude_train_state and asset.kind != 'MODEL':
        raise APIError(422, 'ASSET_TRANSFER_EXCLUDE_MODEL_ONLY', '仅模型资产可以排除根目录下的 train_state/ 子目录')
    if asset.node_id == body.target_node_id:
        raise APIError(422, 'ASSET_TRANSFER_SAME_NODE', '请选择另一台目标机器')
    norm = None
    if body.include_norm_stats:
        # Match asset mutation's asset-before-node order. Holding a node lock
        # while waiting on this NORM could deadlock concurrent archival.
        norm_id = getattr(asset, 'norm_asset_id', None)
        if body.expected_norm_asset_id is not None and body.expected_norm_asset_id != norm_id:
            raise APIError(409, 'ASSET_TRANSFER_NORM_BINDING_CHANGED', '数据集绑定的 norm_stats 已变化，请刷新后重新确认同步')
        norm = session.scalar(select(Asset).where(Asset.id == norm_id).with_for_update()) if norm_id else None
        if (asset.kind != 'DATASET' or norm is None or norm.kind != 'NORM' or norm.node_id != asset.node_id
                or norm.state in {'ARCHIVED', 'UNAVAILABLE'}):
            raise APIError(409, 'ASSET_TRANSFER_NORM_UNAVAILABLE', '一并同步需要数据集已绑定同一源机器上可用的 norm_stats 资产')
    nodes = {n: _online(session, n, settings) for n in sorted({asset.node_id, body.target_node_id})}
    source, target = nodes[asset.node_id], nodes[body.target_node_id]
    for node in (source, target):
        require_not_pending(session, node.id)
    source_path = _path(asset.path, source.roots)
    directory = _path(body.target_directory, target.roots, allow_root=True)
    try:
        target_name = component(body.target_name or PurePosixPath(source_path).name or asset.name)
    except ValueError as exc:
        raise APIError(422, 'ASSET_TRANSFER_NAME_INVALID', str(exc)) from None
    connections = {side: _connection(session, node.id) for side, node in (('source', source), ('target', target))}
    same_machine = (connections['source'].fingerprint == connections['target'].fingerprint
                    or (connections['source'].host, connections['source'].port) ==
                       (connections['target'].host, connections['target'].port))
    source_tree, target_tree = PurePosixPath(source_path), PurePosixPath(directory) / target_name
    if same_machine and (source_tree == target_tree or source_tree in target_tree.parents or target_tree in source_tree.parents):
        raise APIError(422, 'ASSET_TRANSFER_PATH_OVERLAP', '同一台物理机器上的源目录与目标目录不能重叠')
    result = {'source_asset': {'id': asset.id, 'node_id': asset.node_id, 'name': asset.name, 'kind': asset.kind,
                            'path': source_path, 'version': asset.version,
                            'metadata': {key: deepcopy(value) for key, value in asset.metadata_json.items() if key != DATASET_VALIDATION_KEY}},
        'source_roots': deepcopy(source.roots), 'target_roots': deepcopy(target.roots),
        'source_connection_id': connections['source'].id, 'target_connection_id': connections['target'].id,
        'source_connection': _identity(connections['source']), 'target_connection': _identity(connections['target']),
        'target_node_id': target.id, 'target_directory': directory, 'target_name': target_name}
    # Keep false absent so historical snapshots and in-flight retries remain
    # byte-for-byte compatible with the complete-copy contract.
    if body.exclude_train_state:
        result['exclude_train_state'] = True
    if body.include_norm_stats:
        norm_path = _path(norm.path, source.roots)
        if same_machine:
            norm_tree = PurePosixPath(norm_path)
            if norm_tree == target_tree or norm_tree in target_tree.parents or target_tree in norm_tree.parents:
                raise APIError(422, 'ASSET_TRANSFER_PATH_OVERLAP', '目标数据目录不能与绑定的 norm_stats 源目录重叠')
        result['norm_asset'] = {'id': norm.id, 'node_id': norm.node_id, 'name': norm.name, 'kind': norm.kind,
            'path': norm_path, 'version': norm.version, 'metadata': deepcopy(norm.metadata_json)}
    return result


def _check_snapshot(session, row, settings):
    body = AssetTransferInput(source_asset_id=row.source_asset_id, target_node_id=row.target_node_id,
        target_directory=row.target_directory, target_name=row.target_name,
        include_norm_stats=row.include_norm_stats, exclude_train_state=row.exclude_train_state)
    current = _bindings(session, body, settings)
    if current != row.binding_snapshot:
        raise APIError(409, 'ASSET_TRANSFER_BINDING_CHANGED', '源资产、机器目录或 SSH 接入已变化，请新建同步任务')
    return current


def _log(row, message):
    row.logs = (row.logs + f'{now()} {message}\n')[-65536:]
    row.updated_at = now()


def _locked(session, transfer_id, *, include_deleted=False):
    query = select(AssetTransfer).where(AssetTransfer.id == transfer_id)
    if not include_deleted:
        query = query.where(AssetTransfer.deleted_at.is_(None))
    return session.scalar(query.with_for_update().execution_options(populate_existing=True))


def _replay(existing, request_digest):
    if existing.request_digest != request_digest:
        raise APIError(409, 'IDEMPOTENCY_CONFLICT', '同一幂等标识已用于不同的同步请求')
    if existing.deleted_at is not None:
        raise APIError(409, 'ASSET_TRANSFER_DELETED', '该同步任务记录已删除，请重新创建同步任务')
    return asset_transfer_dict(existing)


def _manifest(value):
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'files', 'bytes_total', 'files_total'}
            or type(value.get('schema_version')) is not int or value.get('schema_version') != 1 or not isinstance(value.get('files'), list)
            or len(value['files']) > 100000):
        raise APIError(502, 'ASSET_TRANSFER_INVALID_RESULT', '同步文件清单格式无效')
    paths, total = set(), 0
    for item in value['files']:
        if not isinstance(item, dict) or set(item) != {'path', 'size', 'sha256'}:
            raise APIError(502, 'ASSET_TRANSFER_INVALID_RESULT', '同步文件清单字段无效')
        path = item['path']
        if (not isinstance(path, str) or not path or path == '.' or len(path) > 4096
                or path.startswith('/') or '..' in PurePosixPath(path).parts or str(PurePosixPath(path)) != path
                or path in paths or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in path)
                or type(item['size']) is not int or not 0 <= item['size'] < 2**63
                or not isinstance(item['sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', item['sha256'])):
            raise APIError(502, 'ASSET_TRANSFER_INVALID_RESULT', '同步文件清单包含无效或重复路径')
        paths.add(path)
        total += item['size']
    if (total >= 2**63 or type(value['bytes_total']) is not int or value['bytes_total'] != total
            or type(value['files_total']) is not int or value['files_total'] != len(paths)
            or len(json.dumps(value, ensure_ascii=False).encode()) > 64*1024*1024):
        raise APIError(502, 'ASSET_TRANSFER_INVALID_RESULT', '同步总量与清单不一致或清单过大')
    return value


def _target_available(session, node_id, directory, name, excluding=None):
    target = PurePosixPath(directory) / name
    for row in active_asset_transfers(session, node_id=node_id):
        if row.id == excluding or row.target_node_id != node_id:
            continue
        names = [row.target_name]
        if row.norm_sync:
            names.append(row.norm_sync['target_name'])
        for existing_name in names:
            existing = PurePosixPath(row.target_directory) / existing_name
            if existing == target or existing in target.parents or target in existing.parents:
                raise APIError(409, 'ASSET_TRANSFER_TARGET_BUSY', '该目标目录正在同步，请等待完成或选择其他目录')


def _norm_sync(transfer_id, source):
    return {'source_asset_id': source['id'], 'name': source['name'], 'source_path': source['path'],
        'target_name': 'norm-stats-' + transfer_id, 'target_path': None, 'asset_path': None,
        'target_asset_id': None, 'state': 'QUEUED', 'bytes_total': 0, 'bytes_transferred': 0,
        'files_total': 0, 'files_transferred': 0, 'progress_pct': 0, 'error': None}


def _check_companion_paths(bindings, norm_name):
    source, target = bindings['source_connection'], bindings['target_connection']
    if (source['fingerprint'] != target['fingerprint']
            and (source['host'], source['port']) != (target['host'], target['port'])):
        return
    targets = [PurePosixPath(bindings['target_directory']) / name
               for name in (bindings['target_name'], norm_name)]
    for origin in (bindings['source_asset'], bindings['norm_asset']):
        tree = PurePosixPath(origin['path'])
        if any(tree == dest or tree in dest.parents or dest in tree.parents for dest in targets):
            raise APIError(422, 'ASSET_TRANSFER_PATH_OVERLAP', '同一台物理机器上的源资产与目标数据集、norm_stats 目录不能重叠')


def _stop_norm(row, state, error=None):
    if row.norm_sync and row.norm_sync['state'] != 'SUCCEEDED':
        row.norm_sync = {**row.norm_sync, 'state': state, 'error': deepcopy(error)}


def _validate_result(result, source, directory, name):
    manifest = _manifest(result.get('manifest'))
    if canonical_digest(manifest) != result.get('manifest_sha256'):
        raise APIError(502, 'ASSET_TRANSFER_INVALID_RESULT', '同步返回的文件摘要不一致')
    expected = str(PurePosixPath(directory) / name)
    file_path = str(PurePosixPath(expected) / PurePosixPath(source['path']).name)
    asset_path = result.get('asset_path')
    if (result.get('target_path') != expected or asset_path not in {expected, file_path}
            or result.get('bytes_total') != manifest['bytes_total'] or result.get('files_total') != manifest['files_total']):
        raise APIError(502, 'ASSET_TRANSFER_INVALID_RESULT', '同步结果目录或总量与任务不一致')
    if asset_path != expected and (len(manifest['files']) != 1
            or manifest['files'][0]['path'] != PurePosixPath(file_path).name):
        raise APIError(502, 'ASSET_TRANSFER_INVALID_RESULT', '单文件资产的同步清单与登记路径不一致')
    return manifest, expected, asset_path


def _copy_asset(row, source, asset_path, digest, *, part=None):
    metadata = {k: deepcopy(v) for k, v in source['metadata'].items() if k in SEMANTIC_METADATA}
    metadata['asset_sync'] = {'transfer_id': row.id, 'source_asset_id': source['id'],
        'source_node_id': source['node_id'], 'source_path': source['path'],
        'manifest_sha256': digest, 'byte_verified': True,
        'training_preflight_required': True, 'copied_at': now()}
    if part:
        metadata['asset_sync']['part'] = part
    if row.exclude_train_state and source['kind'] == 'MODEL':
        metadata['asset_sync'].update(exclude_train_state=True, excluded_root_directories=['train_state'])
    return Asset(id=uid(), node_id=row.target_node_id, kind=source['kind'],
        name=source['name'], path=asset_path, version=source['version'], state='REGISTERED',
        metadata_json=metadata, created_at=now())


class AssetTransferManager:
    def __init__(self, db, settings, remote_manager, *, transfer_function=None):
        self.db, self.settings, self.remote_manager = db, settings, remote_manager
        self.transfer_function = transfer_function
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._workers, self._remotes = {}, {}
        self._semaphore = threading.BoundedSemaphore(2)

    def start(self):
        with self.db.write() as session:
            for row in active_asset_transfers(session, lock=True):
                row.state, row.finished_at = 'INTERRUPTED', now()
                row.error = {'code': 'ASSET_TRANSFER_INTERRUPTED', 'message': '控制台已重启，部分文件保留，可重试同步'}
                _stop_norm(row, 'INTERRUPTED', row.error)
                _log(row, row.error['message'])

    def close(self):
        self._stop.set()
        with self._lock:
            remotes = [r for pair in self._remotes.values() for r in pair]
            workers = list(self._workers.values())
        for remote in remotes:
            try:
                remote.close()
            except Exception:
                pass
        with self.db.write() as session:
            for row in active_asset_transfers(session, lock=True):
                row.state, row.finished_at = 'INTERRUPTED', now()
                row.error = {'code': 'ASSET_TRANSFER_INTERRUPTED', 'message': '控制台服务停止，部分文件保留，可重试同步'}
                _stop_norm(row, 'INTERRUPTED', row.error)
                _log(row, row.error['message'])
        for worker in workers:
            worker.join(timeout=5)

    def _launch(self, transfer_id):
        with self.db.read() as session:
            row = session.execute(select(AssetTransfer.state).where(
                AssetTransfer.id == transfer_id, AssetTransfer.deleted_at.is_(None))).first()
            if row is None or row.state not in ASSET_TRANSFER_WORKING:
                return
        with self._lock:
            current = self._workers.get(transfer_id)
            if current and current.is_alive():
                return
            worker = threading.Thread(target=self._run, args=(transfer_id,), name='vla-asset-transfer', daemon=True)
            self._workers[transfer_id] = worker
            worker.start()

    def _cancelled(self, transfer_id, attempt):
        if self._stop.is_set():
            return True
        with self.db.read() as session:
            row = session.execute(select(AssetTransfer.state, AssetTransfer.attempt)
                .where(AssetTransfer.id == transfer_id, AssetTransfer.deleted_at.is_(None))).first()
            return row is None or row.attempt != attempt or row.state not in ASSET_TRANSFER_WORKING

    def _safe(self, session, row, message):
        for side in ('source', 'target'):
            connection = session.get(RemoteConnection, row.binding_snapshot[side + '_connection_id'])
            message = self.remote_manager._safe(message, connection)
        return message

    def _connect_remote(self, remote, transfer_id, attempt):
        try:
            remote.connect()
        except RemoteError as error:
            if error.code != 'SSH_CONNECTION_FAILED':
                raise
            if self._cancelled(transfer_id, attempt):
                return False
            self._progress(transfer_id, attempt, {'stage': 'PREPARING',
                'message': 'SSH 连接暂时不可用，0.5 秒后重试一次；尚未开始传输文件'})
            if self._stop.wait(.5) or self._cancelled(transfer_id, attempt):
                return False
            # SSHRemote closes a failed client and creates a fresh client on
            # connect(). Only connection setup is retried, never file copying.
            remote.connect()
        return not self._cancelled(transfer_id, attempt)

    def _progress(self, transfer_id, attempt, event, *, part='dataset'):
        with self.db.write() as session:
            row = _locked(session, transfer_id)
            if row is None or row.attempt != attempt or row.state not in ASSET_TRANSFER_WORKING:
                return
            stage = event.get('stage')
            state = 'PREPARING' if stage == 'INSPECTING' else stage
            if state in {'PREPARING', 'TRANSFERRING', 'VERIFYING'}:
                row.state = state
            metrics = {}
            for source, target in (('bytes_done', 'bytes_transferred'), ('bytes_total', 'bytes_total'),
                                   ('files_done', 'files_transferred'), ('files_total', 'files_total')):
                value = event.get(source)
                maximum = 100000 if source.startswith('files_') else 2**63 - 1
                if type(value) is int and 0 <= value <= maximum:
                    metrics[target] = value
                elif source == 'files_done' and source in event and value is None:
                    metrics[target] = -1
            for source, target in (('speed_bps', 'speed_bytes_per_second'), ('percent', 'progress_pct')):
                value = event.get(source)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    metrics[target] = min(value, 100) if source == 'percent' else value
            if part == 'norm' and row.include_norm_stats:
                companion = deepcopy(row.norm_sync)
                companion.update({key: value for key, value in metrics.items() if key != 'speed_bytes_per_second'})
                if state in {'PREPARING', 'TRANSFERRING', 'VERIFYING', 'SUCCEEDED'}:
                    companion['state'] = state
                row.norm_sync = companion
                base = row.manifest or {'bytes_total': 0, 'files_total': 0}
                row.bytes_total = base['bytes_total'] + companion['bytes_total']
                row.bytes_transferred = base['bytes_total'] + companion['bytes_transferred']
                row.files_total = base['files_total'] + companion['files_total']
                row.files_transferred = (-1 if companion['files_transferred'] == -1
                    else base['files_total'] + companion['files_transferred'])
                row.progress_pct = min(99, row.bytes_transferred * 100 / row.bytes_total) if row.bytes_total else 0
                row.speed_bytes_per_second = metrics.get('speed_bytes_per_second', 0)
            else:
                for key, value in metrics.items():
                    setattr(row, key, min(99, value) if row.include_norm_stats and key == 'progress_pct' else value)
            if event.get('message'):
                prefix = 'norm_stats：' if part == 'norm' else ''
                _log(row, self._safe(session, row, prefix + str(event['message'])))
            row.updated_at = now()

    def _run(self, transfer_id):
        remotes, acquired, attempt, part = [], False, None, 'dataset'
        try:
            with self.db.read() as session:
                row = session.get(AssetTransfer, transfer_id)
                if row is None or row.deleted_at is not None:
                    return
                attempt = row.attempt
                if row.state not in ASSET_TRANSFER_WORKING:
                    return
            while not self._cancelled(transfer_id, attempt):
                if self._semaphore.acquire(timeout=.2):
                    acquired = True
                    break
            if not acquired:
                return
            with self.db.write() as session:
                row = _locked(session, transfer_id)
                if row is None or row.attempt != attempt or row.state not in ASSET_TRANSFER_WORKING:
                    return
                bindings = _check_snapshot(session, row, self.settings)
                for side in ('source', 'target'):
                    connection = session.get(RemoteConnection, bindings[side + '_connection_id'])
                    remotes.append(self.remote_manager._remote(connection))
                row.state = 'PREPARING'
                _log(row, '检查源资产与目标目录；文件由源机器直传目标机器，控制台只负责调度、校验与进度')
            with self._lock:
                self._remotes[transfer_id] = remotes
            for remote in remotes:
                if self._cancelled(transfer_id, attempt):
                    return
                if not self._connect_remote(remote, transfer_id, attempt):
                    return
            transfer = self.transfer_function
            if transfer is None:
                from .asset_transfer_transport import transfer_asset
                transfer = transfer_asset
            result = transfer(*remotes, bindings['source_asset']['path'], bindings['target_directory'], transfer_id,
                bindings['source_roots'], bindings['target_roots'], target_name=bindings['target_name'],
                progress=lambda event: self._progress(transfer_id, attempt, event),
                cancelled=lambda: self._cancelled(transfer_id, attempt),
                **({'exclude_train_state': True} if bindings.get('exclude_train_state') else {}))
            if self._cancelled(transfer_id, attempt):
                return
            manifest, expected, asset_path = _validate_result(result, bindings['source_asset'],
                bindings['target_directory'], bindings['target_name'])
            norm_source = bindings.get('norm_asset')
            norm_result = norm_manifest = norm_path = None
            if norm_source is not None:
                # The completed dataset remains owned by this transfer. A retry
                # revalidates that published directory rather than overwriting it.
                with self.db.write() as session:
                    row = _locked(session, transfer_id)
                    if row is None or row.attempt != attempt or row.state not in ASSET_TRANSFER_WORKING:
                        return
                    row.manifest, row.manifest_sha256 = deepcopy(manifest), result['manifest_sha256']
                    row.target_path, row.asset_path = expected, asset_path
                    row.state = 'PREPARING'
                    _log(row, '数据集文件已校验，继续同步已绑定的 norm_stats；两份完成后登记目标资产')
                if self._cancelled(transfer_id, attempt):
                    return
                part = 'norm'
                norm_result = transfer(*remotes, norm_source['path'], bindings['target_directory'], transfer_id + '-norm',
                    bindings['source_roots'], bindings['target_roots'], target_name='norm-stats-' + transfer_id,
                    progress=lambda event: self._progress(transfer_id, attempt, event, part='norm'),
                    cancelled=lambda: self._cancelled(transfer_id, attempt))
                if self._cancelled(transfer_id, attempt):
                    return
                norm_manifest, norm_target, norm_path = _validate_result(norm_result, norm_source,
                    bindings['target_directory'], 'norm-stats-' + transfer_id)
            with self.db.write() as session:
                row = _locked(session, transfer_id)
                if row is None or row.attempt != attempt or row.state not in ASSET_TRANSFER_WORKING:
                    return
                _check_snapshot(session, row, self.settings)
                source = row.binding_snapshot['source_asset']
                asset = _copy_asset(row, source, asset_path, result['manifest_sha256'])
                if norm_source is not None:
                    norm = _copy_asset(row, norm_source, norm_path, norm_result['manifest_sha256'], part='norm_stats')
                    session.add(norm)
                    session.flush()
                    asset.norm_asset_id = norm.id
                    row.norm_sync = {**row.norm_sync, 'state': 'SUCCEEDED', 'target_path': norm_target,
                        'asset_path': norm_path, 'target_asset_id': norm.id, 'error': None,
                        'manifest_sha256': norm_result['manifest_sha256'],
                        'bytes_total': norm_manifest['bytes_total'], 'bytes_transferred': norm_manifest['bytes_total'],
                        'files_total': norm_manifest['files_total'], 'files_transferred': norm_manifest['files_total'],
                        'progress_pct': 100}
                session.add(asset)
                session.flush()
                queue_validation(session, asset, heartbeat_timeout_seconds=self.settings.heartbeat_timeout_seconds)
                row.target_path, row.asset_path, row.target_asset_id = expected, asset_path, asset.id
                row.manifest, row.manifest_sha256 = deepcopy(manifest), result['manifest_sha256']
                row.bytes_total = row.bytes_transferred = manifest['bytes_total'] + (norm_manifest['bytes_total'] if norm_manifest else 0)
                row.files_total = row.files_transferred = manifest['files_total'] + (norm_manifest['files_total'] if norm_manifest else 0)
                row.progress_pct, row.speed_bytes_per_second = 100, 0
                row.state, row.error, row.finished_at = 'SUCCEEDED', None, now()
                _log(row, '文件已逐项校验并登记到目标机器；训练前仍需预检数据语义与处理配置')
                if norm_source is not None:
                    _log(row, 'norm_stats 已作为目标机器资产登记，并绑定到同步后的数据集')
        except Exception as exc:
            code, message = _failure_details(exc)
            if attempt is not None:
                with self.db.write() as session:
                    row = _locked(session, transfer_id)
                    if row is not None and row.attempt == attempt and row.state in ASSET_TRANSFER_WORKING:
                        row.state, row.finished_at, row.speed_bytes_per_second = 'FAILED', now(), 0
                        row.error = {'code': code,
                            'message': self._safe(session, row, message)[:2000]}
                        if part == 'norm' and row.norm_sync:
                            row.norm_sync = {**row.norm_sync, 'state': 'FAILED', 'error': deepcopy(row.error)}
                        _log(row, row.error['message'])
                LOG.info('Asset transfer %s failed: %s', transfer_id, code)
        finally:
            for remote in remotes:
                try:
                    remote.close()
                except Exception:
                    pass
            with self._lock:
                self._remotes.pop(transfer_id, None)
            if attempt is not None:
                with self.db.write() as session:
                    row = _locked(session, transfer_id)
                    if row is not None and row.attempt == attempt and row.state == 'CANCELLING':
                        row.state, row.finished_at, row.speed_bytes_per_second = 'CANCELLED', now(), 0
                        _stop_norm(row, 'CANCELLED')
                        _log(row, '同步已取消；本任务的部分文件保留，可重试继续同步')
            if acquired:
                self._semaphore.release()


def register_routes(app, admin, manager):
    db, settings = manager.db, manager.settings

    @app.get('/api/v1/asset-transfers', dependencies=admin)
    def list_transfers(source_asset_id: str | None = Query(default=None, min_length=1, max_length=36),
                       target_node_id: str | None = Query(default=None, min_length=1, max_length=36),
                       limit: int = Query(default=100, ge=1, le=200)):
        with db.read() as session:
            query = select(AssetTransfer).options(defer(AssetTransfer.manifest), defer(AssetTransfer.binding_snapshot))
            query = query.where(AssetTransfer.deleted_at.is_(None)).order_by(AssetTransfer.created_at.desc(), AssetTransfer.id.desc())
            if source_asset_id is not None:
                query = query.where(AssetTransfer.source_asset_id == source_asset_id)
            if target_node_id is not None:
                query = query.where(AssetTransfer.target_node_id == target_node_id)
            return [asset_transfer_dict(row) for row in session.scalars(query.limit(limit))]

    @app.get('/api/v1/asset-transfers/{transfer_id}', dependencies=admin)
    def get_transfer(transfer_id: str):
        with db.read() as session:
            row = session.get(AssetTransfer, transfer_id,
                options=[defer(AssetTransfer.manifest), defer(AssetTransfer.binding_snapshot)])
            if row is None or row.deleted_at is not None:
                raise APIError(404, 'NOT_FOUND', '同步任务不存在')
            return asset_transfer_dict(row)

    @app.delete('/api/v1/asset-transfers/{transfer_id}', dependencies=admin, status_code=204)
    def delete_transfer(transfer_id: str):
        with db.write() as session:
            row = _locked(session, transfer_id, include_deleted=True)
            if row is None:
                raise APIError(404, 'NOT_FOUND', '同步任务不存在')
            if row.deleted_at is None:
                if row.state not in ASSET_TRANSFER_TERMINAL:
                    raise APIError(409, 'ASSET_TRANSFER_NOT_TERMINAL', '只能删除已结束的数据同步记录，请先取消同步并等待执行结束')
                # Keep immutable request keys, receipts, registered assets and
                # norm bindings. Deleting a record never touches machine files.
                row.deleted_at = now()
        return Response(status_code=204)

    @app.post('/api/v1/asset-transfers', dependencies=admin, status_code=202)
    def create_transfer(body: AssetTransferInput):
        key = body.idempotency_key or uid()
        try:
            digest_fields = body.model_dump(exclude={'idempotency_key'})
            if not body.exclude_train_state:
                digest_fields.pop('exclude_train_state')
            if not body.include_norm_stats:
                digest_fields.pop('include_norm_stats')  # Preserve pre-feature idempotency keys.
                digest_fields.pop('expected_norm_asset_id')
            request_digest = canonical_digest(digest_fields)
        except UnicodeError:
            raise APIError(422, 'ASSET_TRANSFER_INVALID_TEXT', '同步请求包含无效文本编码') from None
        try:
            with db.write() as session:
                _admission_lock(session)
                existing = session.scalar(select(AssetTransfer).where(AssetTransfer.request_key == key))
                if existing is not None:
                    return _replay(existing, request_digest)
                bindings = _bindings(session, body, settings)
                if len(active_asset_transfers(session)) >= 20:
                    raise APIError(429, 'ASSET_TRANSFER_BUSY', '当前同步任务较多，请等待已有任务完成')
                _target_available(session, body.target_node_id, bindings['target_directory'], bindings['target_name'])
                _verify_saved_ssh_credentials(session, bindings, manager)
                source = bindings['source_asset']
                transfer_id = uid()
                companion = _norm_sync(transfer_id, bindings['norm_asset']) if body.include_norm_stats else None
                if companion:
                    if companion['target_name'] == bindings['target_name']:
                        raise APIError(422, 'ASSET_TRANSFER_PATH_OVERLAP', '数据集与 norm_stats 目标目录不能相同')
                    _target_available(session, body.target_node_id, bindings['target_directory'], companion['target_name'])
                    _check_companion_paths(bindings, companion['target_name'])
                row = AssetTransfer(id=transfer_id, request_key=key, request_digest=request_digest,
                    name=body.name or (source['name'] + ' · 资产同步')[:200],
                    source_asset_id=source['id'], source_node_id=source['node_id'], source_path=source['path'],
                    source_kind=source['kind'], target_node_id=body.target_node_id,
                    target_directory=bindings['target_directory'], target_name=bindings['target_name'],
                    include_norm_stats=body.include_norm_stats, norm_sync=companion,
                    exclude_train_state=body.exclude_train_state,
                    state='QUEUED', attempt=1, logs='', binding_snapshot=bindings, created_at=now(), updated_at=now())
                session.add(row)
                session.flush()
                transfer_id, result = row.id, asset_transfer_dict(row)
        except IntegrityError:
            with db.read() as session:
                existing = session.scalar(select(AssetTransfer).where(AssetTransfer.request_key == key))
                if existing is not None and existing.request_digest == request_digest:
                    return _replay(existing, request_digest)
            raise APIError(409, 'IDEMPOTENCY_CONFLICT', '同步请求已提交或发生并发冲突，请刷新后重试') from None
        manager._launch(transfer_id)
        return result

    @app.post('/api/v1/asset-transfers/{transfer_id}/retry', dependencies=admin, status_code=202)
    def retry_transfer(transfer_id: str):
        with db.read() as session:
            row = session.get(AssetTransfer, transfer_id)
            if row is None or row.deleted_at is not None:
                raise APIError(404, 'NOT_FOUND', '同步任务不存在')
        with manager._lock:
            worker = manager._workers.get(transfer_id)
            if worker is not None and worker.is_alive():
                raise APIError(409, 'ASSET_TRANSFER_CANCELLING', '上一次同步仍在结束，请稍后重试')
        with db.write() as session:
            _admission_lock(session)
            row = _locked(session, transfer_id)
            if row is None:
                raise APIError(404, 'NOT_FOUND', '同步任务不存在')
            if row.state not in ASSET_TRANSFER_RETRYABLE:
                raise APIError(409, 'ASSET_TRANSFER_NOT_RETRYABLE', '只有失败、取消或中断的同步可以重试')
            if len(active_asset_transfers(session)) >= 20:
                raise APIError(429, 'ASSET_TRANSFER_BUSY', '当前同步任务较多，请等待已有任务完成')
            bindings = _check_snapshot(session, row, settings)
            _target_available(session, row.target_node_id, row.target_directory, row.target_name, excluding=row.id)
            if row.norm_sync:
                _target_available(session, row.target_node_id, row.target_directory, row.norm_sync['target_name'], excluding=row.id)
            _verify_saved_ssh_credentials(session, bindings, manager)
            row.attempt += 1
            row.state, row.error, row.finished_at = 'QUEUED', None, None
            row.bytes_total = row.bytes_transferred = row.files_total = row.files_transferred = 0
            row.progress_pct = row.speed_bytes_per_second = 0
            if row.include_norm_stats:
                row.norm_sync = _norm_sync(row.id, bindings['norm_asset'])
            _log(row, '重新核对源资产与目标目录，并复用本任务已传输的部分文件')
            result = asset_transfer_dict(row)
        manager._launch(transfer_id)
        return result

    @app.post('/api/v1/asset-transfers/{transfer_id}/cancel', dependencies=admin, status_code=202)
    def cancel_transfer(transfer_id: str):
        with manager._lock:
            worker = manager._workers.get(transfer_id)
            alive = worker is not None and worker.is_alive()
        with db.write() as session:
            row = _locked(session, transfer_id)
            if row is None:
                raise APIError(404, 'NOT_FOUND', '同步任务不存在')
            if row.state in ASSET_TRANSFER_ACTIVE:
                row.state = 'CANCELLING' if alive else 'CANCELLED'
                row.speed_bytes_per_second = 0
                row.finished_at = None if alive else now()
                if not alive:
                    _stop_norm(row, 'CANCELLED')
                _log(row, '正在取消同步；源文件和本任务已传输的部分文件均保留')
            result = asset_transfer_dict(row)
        with manager._lock:
            remotes = list(manager._remotes.get(transfer_id, []))
        for remote in remotes:
            try:
                remote.close()
            except Exception:
                pass
        return result
