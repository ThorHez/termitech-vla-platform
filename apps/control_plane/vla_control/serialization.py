from sqlalchemy import select

from .models import Allocation, Asset, Attempt


def node_dict(session, node):
    reserved = {item.gpu_uuid: item.attempt_id for item in session.scalars(select(Allocation).where(Allocation.node_id == node.id))}
    return {"id": node.id, "name": node.effective_name, "hostname": node.hostname, "status": node.status,
            "last_seen": node.last_seen, "roots": node.roots, "output_root": node.output_root,
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
    result["training_stage"] = run.code_expectation.get("training_stage")
    result["stage_config"] = run.code_expectation.get("stage_config")
    result["attempts"] = [attempt_dict(a) for a in attempts]
    ids = [a.id for a in attempts]
    result["outputs"] = [asset_dict(a) for a in session.scalars(select(Asset).where(Asset.origin_attempt_id.in_(ids)))] if ids else []
    result["input_assets"] = [asset_dict(a) for a in session.scalars(select(Asset).where(Asset.id.in_(list(run.inputs.values()))))] if run.inputs else []
    return result
