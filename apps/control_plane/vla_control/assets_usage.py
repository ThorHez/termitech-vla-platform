"""Keep registered checkpoint inputs visible while inference/export uses them."""

from pathlib import PurePosixPath

from sqlalchemy import select

from .checkpoint_transfer_models import CheckpointTransfer, TRANSFER_ACTIVE
from .common import APIError, absolute_path
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES


def _canonical(value):
    # Match safe POSIX spelling only; the controller never resolves remote
    # symlinks or treats a broad ancestor as evidence of a checkpoint binding.
    if not isinstance(value, str) or not absolute_path(value) or any(ord(char) < 32 for char in value):
        return None
    return PurePosixPath('/' + str(PurePosixPath(value)).lstrip('/'))


def _references(asset, checkpoint_path):
    path, checkpoint = _canonical(asset.path), _canonical(checkpoint_path)
    if path is None or checkpoint is None:
        return False
    if asset.kind == 'MODEL':
        # Agent policy_deployment explicitly consumes these weight locations.
        return path in {checkpoint, checkpoint / 'params', checkpoint / 'model.safetensors'}
    assets = checkpoint / 'assets'
    return asset.kind == 'NORM' and (path == assets or assets in path.parents)


def require_no_active_checkpoint_reference(session, asset):
    """Called after AuthGuard -> Asset -> Node locks in catalogue archival.

    Deployment/transfer admission also serializes through AuthGuard, so no new
    human workflow can bind this path between this check and archival. Reads
    avoid taking task locks after Node; workers may finish in parallel and a
    conservative conflict can be retried. Ownership of the consumer is irrelevant.
    """
    if asset.kind not in {'MODEL', 'NORM'}:
        return
    deployments = session.scalars(select(PolicyDeployment.checkpoint_path).where(
        PolicyDeployment.node_id == asset.node_id, PolicyDeployment.state.in_(RESOURCE_STATES)))
    exports = session.scalars(select(CheckpointTransfer.source_checkpoint_path).where(
        CheckpointTransfer.source_node_id == asset.node_id, CheckpointTransfer.state.in_(TRANSFER_ACTIVE)))
    if any(_references(asset, path) for paths in (deployments, exports) for path in paths):
        raise APIError(409, 'ASSET_IN_USE', '此资产正在被推理部署或 checkpoint 同步使用，请等待任务完成或先停止相关服务')
