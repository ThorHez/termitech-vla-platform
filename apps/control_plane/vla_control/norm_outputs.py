"""Publish normalization results against their frozen source asset identities."""

from pathlib import PurePosixPath

from sqlalchemy import select
from vla_adapters.norm_entry import norm_artifact_name, norm_dataset_name

from .common import APIError
from .dataset_norm_bindings import update_norm_binding
from .models import Asset, Attempt, Node


BINDING_EXPECTATIONS = "_norm_binding_expectations"


def norm_source_ids(run):
    if run.template_id != "openpi-norm":
        return []
    rows = run.code_expectation.get("norm_datasets")
    expected = ({"dataset": run.inputs.get("dataset")} if rows is None else
                {f"dataset_{index}": row["dataset_asset_id"] for index, row in enumerate(rows)})
    if run.inputs != expected or not expected or any(not isinstance(value, str) for value in expected.values()):
        raise ValueError("Statistics sources differ from the frozen run inputs")
    return list(expected.values())


def freeze_norm_bindings(session, inputs, node_id):
    sources = {}
    for asset_id in sorted(inputs.values()):
        # Submission already holds the Node lock. Read without an Asset lock
        # to preserve archive/transfer's Asset -> Node lock order. A concurrent
        # binding edit is protected by the completion-time comparison below.
        source = session.scalar(select(Asset).where(Asset.id == asset_id)
                                .execution_options(populate_existing=True))
        if source is None or source.kind != "DATASET" or source.node_id != node_id:
            raise APIError(422, "NORM_DATASETS_MISMATCH", "统计来源与冻结的数据集资产不一致")
        sources[asset_id] = source.norm_asset_id
    return sources


def _source_id(run, metadata):
    if "source_dataset_asset_id" in metadata:
        return metadata["source_dataset_asset_id"]
    # Older single-source Agents did not report this field. The sole frozen
    # input is authoritative; names and filesystem spelling never select it.
    sources = norm_source_ids(run)
    return sources[0] if run.code_expectation.get("norm_datasets") is None and len(sources) == 1 else None


def norm_outputs_error(session, node, run, outputs):
    if run.template_id != "openpi-norm":
        return None
    try:
        expected = norm_source_ids(run)
    except (ValueError, TypeError, KeyError):
        return {"code": "OUTPUT_INVALID", "message": "统计来源与本次冻结输入不一致"}
    if not isinstance(outputs, list) or len(outputs) != len(expected):
        return {"code": "OUTPUT_INCOMPLETE", "message": "统计必须为每个数据集产出独立的 norm_stats"}
    sources, paths = [], []
    for output in outputs:
        if (not isinstance(output, dict) or output.get("kind") != "NORM"
                or not isinstance(output.get("metadata", {}), dict) or not isinstance(output.get("path"), str)):
            return {"code": "OUTPUT_INVALID", "message": "统计产物缺少数据集来源或独立输出路径"}
        source_id = _source_id(run, output.get("metadata", {}))
        if not isinstance(source_id, str) or source_id not in expected:
            return {"code": "OUTPUT_INVALID", "message": "统计产物来源不是本次冻结的数据集"}
        source = session.get(Asset, source_id)
        if source is None or source.kind != "DATASET" or source.node_id != node.id or source.node_id != run.node_id:
            return {"code": "OUTPUT_INVALID", "message": "统计来源必须是本次机器上的数据集资产"}
        sources.append(source_id)
        paths.append(str(PurePosixPath(output["path"])))
    if set(sources) != set(expected) or len(set(sources)) != len(expected) or len(set(paths)) != len(expected):
        return {"code": "OUTPUT_INCOMPLETE", "message": "统计产物与数据集未逐项对应，不能使用共享文件替代独立统计"}
    return None


def reconcile_norm_outputs(session, run, *, attempt_id=None):
    """Idempotently name and bind a successful run's complete output set.

    New runs compare the current binding with the value frozen at submission.
    Historical runs without that snapshot may fill only an empty binding.
    Conflicts retain the valid output and the user's current dataset default.
    The caller owns the transaction; this helper never commits independently.
    """
    if run.template_id != "openpi-norm" or run.state != "SUCCEEDED":
        raise ValueError("Only a successful statistics run can publish dataset bindings")
    query = select(Attempt).where(Attempt.run_id == run.id, Attempt.state == "SUCCEEDED")
    if attempt_id is not None:
        query = query.where(Attempt.id == attempt_id)
    attempt = session.scalar(query.order_by(Attempt.number.desc()).limit(1))
    if attempt is None:
        raise ValueError("Statistics run has no successful attempt")
    outputs = list(session.scalars(select(Asset).where(Asset.origin_attempt_id == attempt.id).order_by(Asset.id)))
    node = session.get(Node, run.node_id)
    manifests = [{"kind": output.kind, "path": output.path, "metadata": output.metadata_json} for output in outputs]
    if node is None or norm_outputs_error(session, node, run, manifests) or any(output.node_id != run.node_id for output in outputs):
        raise ValueError("Statistics output sources do not match the complete frozen input set")
    expected = run.code_expectation.get(BINDING_EXPECTATIONS)
    locked = {asset_id: session.scalar(select(Asset).where(Asset.id == asset_id).with_for_update()
              .execution_options(populate_existing=True)) for asset_id in sorted(norm_source_ids(run))}
    if any(source is None or source.kind != "DATASET" or source.node_id != run.node_id for source in locked.values()):
        raise ValueError("Statistics source identity changed before binding")
    results = []
    for norm in outputs:
        source_id = _source_id(run, norm.metadata_json)
        source = locked[source_id]
        status, reason = "BOUND", None
        original = expected.get(source_id) if isinstance(expected, dict) and source_id in expected else None
        if source.norm_asset_id != norm.id:
            if source.norm_asset_id != original:
                status, reason = "SKIPPED", "BINDING_CHANGED"
            else:
                try:
                    update_norm_binding(session, source.id, norm.id)
                except APIError as exc:
                    status, reason = "SKIPPED", exc.code
        source_description = {"id": source.id, "name": source.name, "path": source.path}
        norm.name = norm_artifact_name(source_description)
        binding = {"status": status, **({"reason": reason} if reason else {})}
        norm.metadata_json = {**norm.metadata_json, "source_dataset_asset_id": source_id,
                              "source_dataset_name": norm_dataset_name(source_description), "dataset_binding": binding}
        results.append({"norm_asset_id": norm.id, "source_dataset_asset_id": source_id,
                        "name": norm.name, "binding": binding})
    session.flush()
    return results
