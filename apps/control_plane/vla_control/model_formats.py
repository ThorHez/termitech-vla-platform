"""Manual model format declarations and compatible legacy format names."""

from sqlalchemy import select

from .common import APIError
from .models import Asset
from .resource_authorization import request_user_id, require_active_actor


ORBAX_FORMAT = "openpi-orbax"
PYTORCH_FORMAT = "openpi-pytorch"
_ORBAX_NAMES = frozenset({ORBAX_FORMAT, "orbax", "jax", "openpi-jax"})
_PYTORCH_NAMES = frozenset({PYTORCH_FORMAT, "torch", "pytorch", "openpi-torch"})


def normalize_model_format(value):
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    if name in _ORBAX_NAMES:
        return ORBAX_FORMAT
    if name in _PYTORCH_NAMES:
        return PYTORCH_FORMAT
    return None


def update_model_format(session, request, asset_id, model_format):
    from .asset_transfers import active_asset_transfers

    require_active_actor(session, request, "assets.register")
    asset = session.scalar(select(Asset).where(Asset.id == asset_id).with_for_update()
                           .execution_options(populate_existing=True))
    if asset is None:
        raise APIError(404, "NOT_FOUND", "Asset not found")
    if asset.kind != "MODEL":
        raise APIError(409, "ASSET_KIND_MISMATCH", "只有模型资产可以编辑模型格式")
    if request.state.principal.role != "admin" and (
            asset.registered_by_user_id is None or asset.registered_by_user_id != request_user_id(request)):
        raise APIError(403, "RESOURCE_ACCESS_DENIED", "只能编辑自己登记的模型；未记录登记人的模型由管理员管理")
    if asset.state not in {"REGISTERED", "READY"}:
        raise APIError(409, "ASSET_UNAVAILABLE", "已归档或不可用的模型资产不能编辑格式")
    metadata = dict(asset.metadata_json or {})
    existing_format = metadata.get("format")
    if (asset.origin_attempt_id is not None or "asset_sync" in metadata or "transfer_id" in metadata
            or existing_format == "openpi-transferred"):
        raise APIError(409, "MODEL_FORMAT_READONLY", "训练或同步生成的模型格式由来源记录确定")
    if model_format is None:
        if normalize_model_format(existing_format) is None:
            return asset
        metadata.pop("format")
    elif model_format == existing_format:
        return asset
    else:
        metadata["format"] = model_format
    if active_asset_transfers(session, asset_id=asset_id):
        raise APIError(409, "ASSET_IN_USE", "资产正在同步，请等待完成或先取消同步")
    asset.metadata_json = metadata
    session.flush()
    return asset
