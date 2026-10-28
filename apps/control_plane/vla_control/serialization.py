import ipaddress

from sqlalchemy import func, select

from .models import Allocation, Asset, Attempt
from .policy_deployment_models import PolicyDeployment, RESOURCE_STATES
from .remote_models import RemoteConnection, RemoteDeployment
from .remote_schemas import SSHAuthenticationInput


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
        if operation not in (None, "install") or type(port) is not int or not 1 <= port <= 65535:
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
    reserved = {item.gpu_uuid: item.attempt_id for item in session.scalars(select(Allocation).where(Allocation.node_id == node.id))}
    for item in session.scalars(select(PolicyDeployment).where(
            PolicyDeployment.node_id == node.id, PolicyDeployment.state.in_(RESOURCE_STATES))):
        if item.gpu_uuid:
            reserved.setdefault(item.gpu_uuid, f"deployment:{item.id}")
    return {"id": node.id, "name": node.effective_name, "hostname": node.hostname, "status": node.status,
            "connection_address": _connection_address(session, node),
            "last_seen": node.last_seen, "roots": node.roots, "output_root": node.output_root, "cache_root": node.cache_root,
            "gpus": [{**gpu, "reserved_by": reserved.get(gpu["uuid"])} for gpu in node.gpus],
            "metrics": node.metrics, "created_at": node.created_at}


def profile_dict(profile):
    return {"id": profile.id, "node_id": profile.node_id, "name": profile.name,
            "code_dir": profile.code_dir, "python": profile.python, "adapter": profile.adapter,
            "recipe": profile.recipe, "execution_mode": profile.execution_mode,
            "docker_image": profile.docker_image, "metadata": profile.metadata_json, "active": profile.active}


def asset_dict(asset):
    return {"id": asset.id, "node_id": asset.node_id, "name": asset.name, "kind": asset.kind,
            "path": asset.path, "version": asset.version, "state": asset.state,
            "metadata": asset.metadata_json, "created_at": asset.created_at,
            "origin_attempt_id": asset.origin_attempt_id}


def preview_dict(preview):
    return {"id": preview.id, "status": preview.status, "checks": preview.checks,
            "code_snapshot": preview.code_snapshot, "error": preview.error, "request": preview.request,
            "created_at": preview.created_at, "expires_at": preview.expires_at}


def attempt_dict(attempt):
    return {key: getattr(attempt, key) for key in ("id", "run_id", "node_id", "number", "state",
            "created_at", "started_at", "finished_at", "pid", "code_snapshot", "gpu_uuids",
            "output_dir", "latest_metrics", "resources", "error", "exit_code", "code_changed_during_run")}


def run_dict(session, run):
    result = {key: getattr(run, key) for key in ("id", "name", "node_id", "profile_id", "template_id",
              "state", "created_at", "updated_at", "finished_at", "inputs", "parameters", "gpu_count",
              "code_expectation", "queue_reason", "error")}
    attempts = list(session.scalars(select(Attempt).where(Attempt.run_id == run.id).order_by(Attempt.number)))
    result["recipe"] = run.code_expectation.get("recipe")
    result["recipe_definition"] = run.code_expectation.get("recipe_definition")
    result["entrypoint"] = run.code_expectation.get("requested_entrypoint")
    result["dataset_bindings"] = run.code_expectation.get("dataset_bindings")
    result["dataset_profiles"] = run.code_expectation.get("dataset_profiles")
    result["training_stage"] = run.code_expectation.get("training_stage")
    result["stage_config"] = run.code_expectation.get("stage_config")
    result["auto_config"] = run.code_expectation.get("auto_config")
    result["confirmed_config_sha256"] = run.code_expectation.get("confirmed_config_sha256")
    result["attempts"] = [attempt_dict(a) for a in attempts]
    ids = [a.id for a in attempts]
    result["outputs"] = [asset_dict(a) for a in session.scalars(select(Asset).where(Asset.origin_attempt_id.in_(ids)))] if ids else []
    result["input_assets"] = [asset_dict(a) for a in session.scalars(select(Asset).where(Asset.id.in_(list(run.inputs.values()))))] if run.inputs else []
    return result
