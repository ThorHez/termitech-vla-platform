import ipaddress
import math
from typing import Literal

from sqlalchemy import func, or_, select
from sqlalchemy.orm import aliased
from vla_adapters.recap import STAGE_ADVANTAGE_STAGES, VALUE_ANNOTATION

from .models import Allocation, Asset, Attempt, Run
from .stage_annotation_sources import stage_annotation_sources
from .network_metrics import PROCESS_METRICS_KEY, network_snapshot
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES
from .remote_models import RemoteConnection, RemoteDeployment
from .remote_schemas import SSHAuthenticationInput
from .gpu_reservations import reservation_dicts


AssetListView = Literal["full", "summary"]
RunListView = Literal["full", "summary"]

ASSET_VALIDATION_SUMMARY_FIELDS = (
    "id", "scope", "status", "profile_id", "requested_at", "checked_at", "completed_at",
    "reason", "error", "source_fingerprint",
)
NODE_ASSET_PREVIEW_COLUMNS = (
    Asset.id, Asset.name, Asset.kind, Asset.node_id, Asset.path,
    Asset.version, Asset.state, Asset.created_at, Asset.norm_asset_id,
)


def _connection_address(session, node):
    # Read only the public connection coordinates, never credentials or probe
    # output. A failed attempt to move a machine must not replace its last
    # successful installation address.
    rows = session.execute(select(RemoteConnection.host, RemoteConnection.port,
                                  RemoteDeployment.agent_config["_operation"].as_string())
        .join(RemoteDeployment, RemoteDeployment.connection_id == RemoteConnection.id)
        .where(RemoteDeployment.node_id == node.id, RemoteDeployment.state == "SUCCEEDED")
        .order_by(func.coalesce(RemoteDeployment.finished_at, RemoteDeployment.created_at).desc(),
                  RemoteDeployment.created_at.desc(), RemoteDeployment.id.desc()))
    for host, port, operation in rows:
        if operation not in (None, "install", "upgrade") or type(port) is not int or not 1 <= port <= 65535:
            continue
        try:
            host = SSHAuthenticationInput.plain_host(host)
        except (AttributeError, TypeError, ValueError):
            continue
        return {"host": host, "port": port, "source": "ssh"}
    try:
        host = str(ipaddress.ip_address(node.hostname))
    except (TypeError, ValueError):
        return None
    return {"host": host, "port": None, "source": "hostname"}


def node_dict(session, node):
    user_reservations = {item["gpu_uuid"]: item for item in reservation_dicts(session, node.id)}
    reserved = {item.gpu_uuid: item.attempt_id for item in session.scalars(select(Allocation).where(Allocation.node_id == node.id))}
    for item in session.scalars(select(PolicyDeployment).where(
            PolicyDeployment.node_id == node.id, PolicyDeployment.state.in_(RESOURCE_STATES))):
        if item.gpu_uuid:
            reserved.setdefault(item.gpu_uuid, f"deployment:{item.id}")
    metrics = node.metrics
    if "network" in metrics or PROCESS_METRICS_KEY in metrics:
        network = network_snapshot(metrics)
        metrics = {key: value for key, value in metrics.items() if key not in ("network", PROCESS_METRICS_KEY)}
        if network is not None:
            metrics["network"] = network
    return {"id": node.id, "name": node.effective_name, "hostname": node.hostname, "status": node.status,
            "connection_address": _connection_address(session, node),
            "last_seen": node.last_seen, "roots": node.roots, "output_root": node.output_root, "cache_root": node.cache_root,
            "gpus": [{**gpu, "reserved_by": reserved.get(gpu["uuid"]),
                      "user_reservation": user_reservations.get(gpu["uuid"])} for gpu in node.gpus],
            "metrics": metrics, "created_at": node.created_at}


def profile_dict(profile):
    return {"id": profile.id, "node_id": profile.node_id, "name": profile.name,
            "code_dir": profile.code_dir, "python": profile.python, "adapter": profile.adapter,
            "recipe": profile.recipe, "execution_mode": profile.execution_mode,
            "docker_image": profile.docker_image, "metadata": profile.metadata_json, "active": profile.active}


def asset_dict(asset, *, summary=False, stage_annotation_source=None, value_annotation_source=None):
    from .dataset_validations import INTERNAL_KEY, validation_dict
    hidden_metadata = {INTERNAL_KEY, "_image_registration"} if summary else {INTERNAL_KEY}
    metadata = {key: value for key, value in (asset.metadata_json or {}).items() if key not in hidden_metadata}
    validation = validation_dict(asset) if asset.kind == "DATASET" else None
    if summary and validation is not None:
        contract = validation.get("timestamp_contract")
        validation = {key: validation[key] for key in ASSET_VALIDATION_SUMMARY_FIELDS if key in validation}
        if isinstance(contract, dict):
            validation["timestamp_contract"] = {key: contract[key] for key in ("status", "checked_rows") if key in contract}
    return {"id": asset.id, "node_id": asset.node_id, "name": asset.name, "kind": asset.kind,
            "path": asset.path, "version": asset.version, "state": asset.state,
            "metadata": metadata,
            **({"stage_annotation_source": stage_annotation_source} if stage_annotation_source is not None else {}),
            **({"value_annotation_source": value_annotation_source} if value_annotation_source is not None else {}),
            **({"validation": validation} if asset.kind == "DATASET" else {}), "created_at": asset.created_at,
            "norm_asset_id": asset.norm_asset_id if asset.kind in {"DATASET", "MODEL"} else None,
            "origin_attempt_id": asset.origin_attempt_id,
            "registered_by_user_id": asset.registered_by_user_id}


def recent_node_assets(session, node_id, limit):
    """Return visible catalog rows and normalization references for a node preview."""
    dataset = aliased(Asset)
    bound_norm = select(dataset.id).where(
        dataset.node_id == node_id, dataset.kind == "DATASET", dataset.state != "ARCHIVED",
        dataset.norm_asset_id == Asset.id,
    ).correlate(Asset).exists()
    assets_query = select(*NODE_ASSET_PREVIEW_COLUMNS).where(
        Asset.node_id == node_id, Asset.state != "ARCHIVED", or_(Asset.kind != "NORM", ~bound_norm),
    ).order_by(Asset.created_at.desc(), Asset.id.desc()).limit(limit)
    assets = [dict(row) for row in session.execute(assets_query).mappings()]
    norm_ids = {asset["norm_asset_id"] for asset in assets if asset["kind"] == "DATASET" and asset["norm_asset_id"]}
    norms = []
    if norm_ids:
        norm_query = select(*NODE_ASSET_PREVIEW_COLUMNS).where(
            Asset.node_id == node_id, Asset.kind == "NORM", Asset.id.in_(norm_ids),
        ).order_by(Asset.id)
        norms = [dict(row) for row in session.execute(norm_query).mappings()]
    return assets, norms


def preview_dict(preview):
    return {"id": preview.id, "owner_user_id": preview.owner_user_id, "status": preview.status, "checks": preview.checks,
            "code_snapshot": preview.code_snapshot, "error": preview.error, "request": preview.request,
            "created_at": preview.created_at, "expires_at": preview.expires_at}


def attempt_dict(attempt):
    return {key: getattr(attempt, key) for key in ("id", "run_id", "node_id", "number", "state",
            "created_at", "started_at", "finished_at", "pid", "code_snapshot", "gpu_uuids",
            "output_dir", "latest_metrics", "resources", "error", "exit_code", "code_changed_during_run")}


def _numeric_fields(**fields):
    return {name: value for name, value in fields.items()
            if type(value) is int or (type(value) is float and math.isfinite(value))}


def _dataset_ids(inputs, bindings):
    ids = {}
    if isinstance(inputs, dict):
        for role, value in inputs.items():
            if role == "dataset" or role.startswith("dataset_"):
                asset_id = value.get("id") if isinstance(value, dict) else value
                if isinstance(asset_id, str) and asset_id:
                    ids[asset_id] = None
    if isinstance(bindings, list):
        for binding in bindings:
            asset_id = binding.get("dataset_asset_id") if isinstance(binding, dict) else None
            if isinstance(asset_id, str) and asset_id:
                ids[asset_id] = None
    return list(ids)


def run_summaries(session, *, node_id=None, run_ids=None, limit=500):
    """Read only the fields used by run lists, progress, and asset lineage."""
    query = select(
        Run.id, Run.name, Run.node_id, Run.profile_id, Run.template_id, Run.state,
        Run.owner_user_id, Run.created_at, Run.updated_at, Run.finished_at, Run.gpu_count,
        Run.inputs, Run.parameters["steps"].label("steps"),
        Run.code_expectation["dataset_bindings"].label("bindings"),
        Run.code_expectation["requested_entrypoint"].as_string().label("entrypoint"),
    ).where(Run.deleted_at.is_(None))
    if node_id is not None:
        query = query.where(Run.node_id == node_id)
    if run_ids is not None:
        query = query.where(Run.id.in_(run_ids))
    summaries = {}
    for row in session.execute(query.order_by(Run.created_at.desc(), Run.id.desc()).limit(limit)).mappings():
        summary = dict(row)
        bindings = summary.pop("bindings")
        entrypoint = summary.pop("entrypoint")
        summary["dataset_asset_ids"] = _dataset_ids(summary.pop("inputs"), bindings)
        summary["resume_candidate"] = (
            summary["template_id"] == "openpi-sft" and not entrypoint
            and (not isinstance(bindings, list) or len(bindings) <= 1)
        )
        summary["parameters"] = _numeric_fields(steps=summary.pop("steps"))
        summary["attempts"] = []
        summaries[summary["id"]] = summary
    if not summaries:
        return []

    attempts = select(
        Attempt.run_id, Attempt.id, Attempt.state, Attempt.code_changed_during_run,
        Attempt.latest_metrics["step"].label("step"),
        Attempt.latest_metrics["total_steps"].label("total_steps"),
    ).where(Attempt.run_id.in_(summaries)).order_by(Attempt.run_id, Attempt.number)
    metrics = {}
    for row in session.execute(attempts):
        summaries[row.run_id]["attempts"].append({
            "id": row.id, "state": row.state, "code_changed_during_run": row.code_changed_during_run,
        })
        metrics[row.run_id] = _numeric_fields(step=row.step, total_steps=row.total_steps)
    for run_id, latest_metrics in metrics.items():
        if latest_metrics:
            summaries[run_id]["attempts"][-1]["latest_metrics"] = latest_metrics
    return list(summaries.values())


def run_dict(session, run):
    result = {key: getattr(run, key) for key in ("id", "name", "node_id", "profile_id", "template_id",
              "owner_user_id",
              "state", "created_at", "updated_at", "finished_at", "inputs", "parameters", "gpu_count", "requested_gpu_uuids",
              "code_expectation", "queue_reason", "error")}
    attempts = list(session.scalars(select(Attempt).where(Attempt.run_id == run.id).order_by(Attempt.number)))
    result["recipe"] = run.code_expectation.get("recipe")
    result["recipe_definition"] = run.code_expectation.get("recipe_definition")
    result["entrypoint"] = run.code_expectation.get("requested_entrypoint")
    result["dataset_bindings"] = run.code_expectation.get("dataset_bindings")
    result["norm_datasets"] = run.code_expectation.get("norm_datasets")
    result["dataset_profiles"] = run.code_expectation.get("dataset_profiles")
    result["training_stage"] = run.code_expectation.get("training_stage")
    result["stage_config"] = run.code_expectation.get("stage_config")
    result["auto_config"] = run.code_expectation.get("auto_config")
    if run.code_expectation.get("normalization_method") is not None:
        result["normalization_method"] = run.code_expectation["normalization_method"]
    if run.code_expectation.get("training_gpu_policy") is not None:
        result["training_gpu_policy"] = run.code_expectation["training_gpu_policy"]
    if run.code_expectation.get("resume_source") is not None:
        result["resume"] = True
        result["resume_source"] = run.code_expectation["resume_source"]
    result["confirmed_config_sha256"] = run.code_expectation.get("confirmed_config_sha256")
    result["attempts"] = [attempt_dict(a) for a in attempts]
    ids = [a.id for a in attempts]
    outputs = list(session.scalars(select(Asset).where(Asset.origin_attempt_id.in_(ids)))) if ids else []
    scorer_sources = stage_annotation_sources(session, outputs)
    value_sources = stage_annotation_sources(session, outputs, scorer_stage="value")
    result["outputs"] = [asset_dict(asset, stage_annotation_source=scorer_sources.get(asset.id),
                                    value_annotation_source=value_sources.get(asset.id)) for asset in outputs]
    result["input_assets"] = []
    if run.inputs:
        inputs = select(Asset, Attempt.run_id).outerjoin(Attempt, Asset.origin_attempt_id == Attempt.id).where(
            Asset.id.in_(list(run.inputs.values())))
        for asset, origin_run_id in session.execute(inputs):
            payload = asset_dict(asset)
            metadata = payload["metadata"]
            if origin_run_id and (
                (asset.kind == "DATASET" and metadata.get("registration_source") in
                 ("recap_stage_progress", "recap_stage_annotation", "recap_value_annotation"))
                or (asset.kind == "MODEL" and result["training_stage"] in STAGE_ADVANTAGE_STAGES
                    and metadata.get("model_role") == "stage_progress_estimator")
                or (asset.kind == "MODEL" and result["training_stage"] == VALUE_ANNOTATION
                    and metadata.get("model_role") == "value_estimator")
            ):
                payload["origin_run_id"] = origin_run_id
            result["input_assets"].append(payload)
    return result


def run_status_dict(session, run):
    latest = session.execute(select(
        Attempt.id, Attempt.latest_metrics, Attempt.resources,
    ).where(Attempt.run_id == run.id).order_by(Attempt.number.desc()).limit(1)).mappings().one_or_none()
    attempt = None
    if latest is not None:
        metrics, resources = latest["latest_metrics"] or {}, latest["resources"] or {}
        attempt = {
            "id": latest["id"],
            "latest_metrics": {**_numeric_fields(**metrics), "timestamp": metrics.get("timestamp")},
            "resources": {key: resources[key] for key in ("rss_bytes", "cpu_seconds", "timestamp")
                          if key in resources},
        }
    return {"id": run.id, "state": run.state, "updated_at": run.updated_at,
            "queue_reason": run.queue_reason, "attempt": attempt}
