"""Optional dataset defaults; frozen training and transfer snapshots stay authoritative."""

import hashlib
from pathlib import PurePosixPath

from sqlalchemy import select, text

from .common import APIError, now, require_path, uid
from .models import Asset, Node
from .resource_authorization import require_dataset_registrar


def canonical_norm_path(value):
    """Normalize remote POSIX spelling without resolving files on the controller."""
    if (not isinstance(value, str) or not value.strip() or len(value) > 4096
            or "\\" in value or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)):
        raise APIError(422, "NORM_PATH_INVALID", "请填写有效的 norm_stats 本机绝对路径")
    value = value.strip()
    # Reject traversal before PurePosixPath can remove redundant separators or
    # dot components. A directory and its norm_stats.json file remain distinct:
    # only the Agent can prove their actual filesystem relationship.
    require_path(value)
    canonical = "/" + str(PurePosixPath(value)).lstrip("/")
    if canonical == "/":
        raise APIError(422, "NORM_PATH_INVALID", "请选择 norm_stats 文件或其目录，不能使用文件系统根目录")
    try:
        canonical.encode("utf-8")
    except UnicodeError:
        raise APIError(422, "NORM_PATH_INVALID", "norm_stats 路径包含无效文本") from None
    return canonical


def _lock_norm_path(session, node_id, path):
    # A Node FOR UPDATE here would invert archive/transfer's Asset -> Node
    # order, including the implicit KEY SHARE lock of a norm_asset_id FK write.
    # Serialize automatic registrations only, without taking a Node lock.
    # SQLite Database.write already holds BEGIN IMMEDIATE.
    if session.get_bind().dialect.name == "postgresql":
        identity = ("dataset-norm-path\0" + node_id + "\0" + path).encode("utf-8")
        key = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big", signed=True)
        session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


def resolve_norm_path(session, *, node_id, path, dataset_name, preferred_id=None):
    """Reuse or register one same-machine NORM in the caller's transaction.

    This is metadata registration, not a filesystem or statistics check. The
    caller must commit the dataset and binding in the same transaction, so a
    later validation failure also rolls back a newly created NORM.
    """
    canonical = canonical_norm_path(path)
    node = session.get(Node, node_id)
    if node is None:
        raise APIError(404, "NOT_FOUND", "机器不存在")
    require_path(canonical, node.roots)
    _lock_norm_path(session, node_id, canonical)
    candidates = []
    for norm in session.scalars(select(Asset).where(Asset.node_id == node_id, Asset.kind == "NORM")
                               .order_by(Asset.created_at, Asset.id)):
        try:
            if canonical_norm_path(norm.path) == canonical:
                candidates.append(norm)
        except APIError:
            # A historical malformed asset is not authority for a new path.
            continue
    if preferred_id:
        candidates.sort(key=lambda norm: norm.id != preferred_id)
    for candidate in candidates:
        norm = session.scalar(select(Asset).where(Asset.id == candidate.id).with_for_update()
                              .execution_options(populate_existing=True))
        if (norm is not None and norm.kind == "NORM" and norm.node_id == node_id
                and norm.state not in {"ARCHIVED", "UNAVAILABLE"}
                and canonical_norm_path(norm.path) == canonical):
            return norm
    if candidates:
        raise APIError(409, "NORM_ASSET_UNAVAILABLE", "此路径的 norm_stats 登记已归档或不可用，请先确认已有资产或选择其他路径")
    label = " ".join(str(dataset_name or "").split())
    label = "".join(char for char in label if not (ord(char) < 32 or 127 <= ord(char) <= 159
                                                 or 0xD800 <= ord(char) <= 0xDFFF))
    suffix = " · norm_stats"
    name = label[:200 - len(suffix)] + suffix if label else "norm_stats"
    norm = Asset(id=uid(), node_id=node_id, name=name, kind="NORM", path=canonical,
                 version="v1", state="REGISTERED", created_at=now(),
                 metadata_json={"registration_source": "dataset_norm_binding"})
    session.add(norm)
    session.flush()
    return norm


def validate_norm_binding(session, *, kind, node_id, norm_asset_id):
    """Validate an explicitly selected registered NORM without reading files."""
    if norm_asset_id is None:
        return None
    if kind != "DATASET":
        raise APIError(409, "ASSET_KIND_MISMATCH", "只有数据集资产可以绑定 norm_stats")
    norm = session.scalar(select(Asset).where(Asset.id == norm_asset_id).with_for_update())
    if norm is None:
        raise APIError(404, "NORM_ASSET_NOT_FOUND", "所选 norm_stats 资产不存在")
    if norm.kind != "NORM":
        raise APIError(409, "NORM_ASSET_KIND_MISMATCH", "请选择归一化参数（NORM）资产")
    if norm.node_id != node_id:
        raise APIError(409, "NORM_ASSET_NODE_MISMATCH", "数据集只能绑定同一台机器上的 norm_stats")
    if norm.state in {"ARCHIVED", "UNAVAILABLE"}:
        raise APIError(409, "NORM_ASSET_UNAVAILABLE", "所选 norm_stats 已归档或不可用，请选择其他资产")
    return norm


def update_norm_binding(session, asset_id, norm_asset_id=None, *, norm_path=None, actor_request=None):
    from .asset_transfers import active_asset_transfers

    dataset = session.scalar(select(Asset).where(Asset.id == asset_id).with_for_update())
    if dataset is None:
        raise APIError(404, "NOT_FOUND", "数据集资产不存在")
    if dataset.kind != "DATASET":
        raise APIError(409, "ASSET_KIND_MISMATCH", "只有数据集资产可以绑定 norm_stats")
    if actor_request is not None:
        require_dataset_registrar(actor_request, dataset)
    if dataset.state in {"ARCHIVED", "UNAVAILABLE"}:
        raise APIError(409, "DATASET_ASSET_UNAVAILABLE", "数据集已归档或不可用，不能修改 norm_stats 绑定")
    if norm_path is not None:
        if norm_asset_id is not None:
            raise APIError(422, "NORM_BINDING_INVALID", "请选择已有 norm_stats 或填写路径，不能同时提交两者")
        norm = resolve_norm_path(session, node_id=dataset.node_id, path=norm_path,
                                 dataset_name=dataset.name, preferred_id=dataset.norm_asset_id)
        norm_asset_id = norm.id
    else:
        validate_norm_binding(session, kind=dataset.kind, node_id=dataset.node_id, norm_asset_id=norm_asset_id)
    if dataset.norm_asset_id != norm_asset_id and active_asset_transfers(session, asset_id=dataset.id):
        raise APIError(409, "ASSET_IN_USE", "数据集正在同步，请等待完成或先取消同步再修改 norm_stats 绑定")
    dataset.norm_asset_id = norm_asset_id
    session.flush()
    return dataset
