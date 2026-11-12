"""Record the provenance of an independently annotated ReCap dataset."""

from pathlib import PurePosixPath

from pydantic import Field
from sqlalchemy import select
from vla_adapters.recap_checkpoint import RECAP_TORCH_FORMAT, recap_torch_artifact_valid

from .common import APIError, require_path, under_root
from .dataset_profiles import RECAP_ANNOTATION_SOURCE
from .models import Asset, Attempt, Node
from .resource_authorization import request_user_id, require_active_actor
from .run_records import require_visible_run
from .schemas import InputModel


class RecapAnnotationInput(InputModel):
    checkpoint_asset_id: str = Field(min_length=1, max_length=36, strict=True)
    source_dataset_asset_id: str = Field(min_length=1, max_length=36, strict=True)
    source_dataset_index: int | None = Field(default=None, ge=0, le=63, strict=True)
    dataset_asset_id: str = Field(min_length=1, max_length=36, strict=True)


def _source_rows(run):
    """Return the ordered data/norm identities in the frozen launch inputs."""
    bindings = run.code_expectation.get("dataset_bindings")
    rows = {}
    if bindings is None:
        for role, dataset in run.inputs.items():
            index_text = role.removeprefix("dataset_")
            if role != "dataset" and (not role.startswith("dataset_") or not index_text.isascii()
                                      or not index_text.isdigit() or len(index_text) > 2):
                continue
            index = 0 if role == "dataset" else int(index_text)
            norm = run.inputs.get("norm" + role.removeprefix("dataset"))
            if dataset and norm and index <= 63:
                rows[index] = (dataset, norm)
        return rows
    if not isinstance(bindings, list):
        raise APIError(409, "RECAP_ANNOTATION_SOURCE_INVALID", "原任务的数据和归一化绑定记录不一致")
    for index, row in enumerate(bindings):
        dataset, norm = run.inputs.get(f"dataset_{index}"), run.inputs.get(f"norm_{index}")
        if (not dataset or not norm or not isinstance(row, dict)
                or row.get("dataset_asset_id") != dataset or row.get("norm_asset_id") != norm):
            raise APIError(409, "RECAP_ANNOTATION_SOURCE_INVALID", "原任务的数据和归一化绑定记录不一致")
        rows[index] = (dataset, norm)
    return rows


def _select_source_row(body, rows):
    if body.source_dataset_index is not None:
        row = rows.get(body.source_dataset_index)
        if row is None or row[0] != body.source_dataset_asset_id:
            raise APIError(409, "RECAP_ANNOTATION_SOURCE_INVALID", "所选源数据集与原任务的绑定行不一致")
        return body.source_dataset_index, row[1]
    matches = [(index, norm) for index, (dataset, norm) in rows.items() if dataset == body.source_dataset_asset_id]
    if len({norm for _, norm in matches}) != 1:
        raise APIError(409, "RECAP_ANNOTATION_SOURCE_INVALID", "请选择原任务绑定的源数据集及数据组")
    index, norm = matches[0]
    return (index if len(matches) == 1 else None), norm


def _checkpoint_attempt(session, run, checkpoint, node):
    attempt = session.get(Attempt, checkpoint.origin_attempt_id) if checkpoint and checkpoint.origin_attempt_id else None
    metadata = (checkpoint.metadata_json or {}) if checkpoint else {}
    if (checkpoint is None or checkpoint.kind != "MODEL" or checkpoint.state != "READY" or checkpoint.node_id != node.id
            or metadata.get("availability") == "removed"
            or attempt is None or attempt.run_id != run.id or attempt.node_id != node.id
            or attempt.state != "SUCCEEDED" or not attempt.finished_at or attempt.exit_code != 0
            or not attempt.output_dir or not under_root(checkpoint.path, [attempt.output_dir])
            or PurePosixPath(checkpoint.path) == PurePosixPath(attempt.output_dir)
            or not under_root(checkpoint.path, node.roots)):
        raise APIError(409, "RECAP_ANNOTATION_CHECKPOINT_INVALID", "请选择本次价值训练成功产出的可用检查点")
    checkpoint_format = metadata.get("format")
    if checkpoint_format == RECAP_TORCH_FORMAT:
        plan = run.code_expectation.get("automatic_preflight", {}).get("plan", {})
        descriptor = {"kind": "MODEL", "path": checkpoint.path, "version": checkpoint.version, "metadata": metadata}
        if recap_torch_artifact_valid(descriptor, attempt.output_dir, plan.get("recipe") or run.code_expectation.get("recipe"), "value"):
            return attempt
    elif checkpoint_format == "openpi-orbax":
        step = metadata.get("step")
        if type(step) is int and step >= 0 and checkpoint.version == str(step):
            return attempt
    raise APIError(409, "RECAP_ANNOTATION_CHECKPOINT_INVALID", "所选模型缺少有效的价值检查点格式或步数")


def link_recap_annotation(session, run_id, body: RecapAnnotationInput, request):
    require_active_actor(session, request, "assets.register")
    run = require_visible_run(session, run_id, lock=True)
    if run.state != "SUCCEEDED" or run.template_id != "recap" or run.code_expectation.get("training_stage") != "value":
        raise APIError(409, "RECAP_ANNOTATION_RUN_INVALID", "只可为已成功完成的 ReCap 价值训练关联标注数据")
    rows = _source_rows(run)
    source_index, source_norm_id = _select_source_row(body, rows)
    source_ids = {dataset for dataset, _ in rows.values()}
    identifiers = {*source_ids, body.checkpoint_asset_id, body.dataset_asset_id, source_norm_id}
    assets = {asset.id: asset for asset in session.scalars(
        select(Asset).where(Asset.id.in_(identifiers)).order_by(Asset.id).with_for_update()
        .execution_options(populate_existing=True))}
    target = assets.get(body.dataset_asset_id)
    if target is None:
        raise APIError(404, "NOT_FOUND", "标注数据集资产不存在")
    if request.state.principal.role != "admin" and (
            target.registered_by_user_id is None or target.registered_by_user_id != request_user_id(request)):
        raise APIError(403, "RESOURCE_ACCESS_DENIED", "只能关联自己登记的数据集资产")
    node = session.get(Node, run.node_id)
    if target.kind != "DATASET" or target.node_id != node.id or target.state not in {"REGISTERED", "READY"}:
        raise APIError(409, "RECAP_ANNOTATION_DATASET_INVALID", "请选择同一机器已登记且可用的标注数据集")
    source = assets.get(body.source_dataset_asset_id)
    norm = assets.get(source_norm_id)
    if (source is None or source.kind != "DATASET" or source.node_id != node.id
            or norm is None or norm.kind != "NORM" or norm.node_id != node.id or norm.state not in {"REGISTERED", "READY"}):
        raise APIError(409, "RECAP_ANNOTATION_SOURCE_INVALID", "源数据集或原任务所用归一化资产不可用")
    for asset in (source, norm, target):
        require_path(asset.path, node.roots)
    target_path = "/" + str(PurePosixPath(target.path)).lstrip("/")
    for source_id in source_ids:
        source_asset = assets.get(source_id)
        if source_asset is None:
            continue
        source_path = "/" + str(PurePosixPath(source_asset.path)).lstrip("/")
        if under_root(target_path, [source_path]) or under_root(source_path, [target_path]):
            raise APIError(409, "RECAP_ANNOTATION_IN_PLACE", "标注数据必须使用与训练源数据互不重叠的独立目录")
    checkpoint = assets.get(body.checkpoint_asset_id)
    attempt = _checkpoint_attempt(session, run, checkpoint, node)
    provenance = {"run_id": run.id, "attempt_id": attempt.id, "checkpoint_asset_id": checkpoint.id,
                  "source_dataset_asset_id": source.id, "source_norm_asset_id": norm.id}
    metadata = target.metadata_json
    if RECAP_ANNOTATION_SOURCE in metadata:
        recorded = metadata[RECAP_ANNOTATION_SOURCE]
        recorded_index = recorded.get("source_dataset_index") if isinstance(recorded, dict) else None
        if (not isinstance(recorded, dict)
                or {key: value for key, value in recorded.items() if key != "source_dataset_index"} != provenance
                or recorded_index is not None and source_index is not None and recorded_index != source_index):
            raise APIError(409, "RECAP_ANNOTATION_CONFLICT", "该标注数据集已关联其他价值训练、检查点或源数据组")
        if source_index is None or recorded_index == source_index:
            return target
    if metadata.get("registration_source") == "recap_stage_progress":
        raise APIError(409, "RECAP_ANNOTATION_CONFLICT", "该数据集已有 Kai0 优势标注来源")
    if source_index is not None:
        provenance["source_dataset_index"] = source_index
    target.metadata_json = {**metadata, RECAP_ANNOTATION_SOURCE: provenance}
    session.flush()
    return target
