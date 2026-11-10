"""Bind an SFT resume to its completed, same-node platform output."""

from pathlib import PurePosixPath

from vla_protocol import canonical_digest
from vla_protocol.training_resume import validate_resume_evidence, validate_resume_source

from .auth_models import User
from .common import APIError, TERMINAL, snapshot_identity, under_root, valid_snapshot
from .models import Asset, Attempt, Run
from .resource_authorization import require_resource_owner


def _source(session, node, asset_id):
    asset = session.get(Asset, asset_id) if asset_id else None
    attempt = session.get(Attempt, asset.origin_attempt_id) if asset and asset.origin_attempt_id else None
    run = session.get(Run, attempt.run_id) if attempt else None
    if (asset is None or asset.kind != "MODEL" or asset.state != "READY" or asset.node_id != node.id
            or attempt is None or attempt.node_id != node.id or attempt.state not in TERMINAL
            or attempt.started_at is None or attempt.finished_at is None or attempt.code_changed_during_run is True
            or run is None or run.node_id != node.id or run.deleted_at is not None or run.state not in TERMINAL
            or run.template_id != "openpi-sft"):
        raise APIError(409, "SFT_RESUME_SOURCE_UNAVAILABLE", "请选择本机已结束 SFT 任务的可用检查点")

    metadata = asset.metadata_json if isinstance(asset.metadata_json, dict) else {}
    descriptor = {"asset_id": asset.id, "run_id": run.id, "attempt_id": attempt.id,
                  "path": asset.path, "output_dir": attempt.output_dir, "step": metadata.get("step")}
    try:
        validate_resume_source(descriptor)
    except ValueError as exc:
        raise APIError(409, "SFT_RESUME_SOURCE_UNAVAILABLE", "检查点路径、步数或产出归属无效") from exc
    if (metadata.get("format") != "openpi-orbax" or asset.version != str(descriptor["step"])
            or not under_root(descriptor["output_dir"], [node.output_root])
            or not under_root(descriptor["path"], node.roots)):
        raise APIError(409, "SFT_RESUME_SOURCE_UNAVAILABLE", "检查点不是本机平台产出的 Orbax 训练检查点")

    snapshot, expectation = attempt.code_snapshot, run.code_expectation
    if (not valid_snapshot(snapshot) or not valid_snapshot(expectation)
            or snapshot_identity(snapshot) != snapshot_identity(expectation)):
        raise APIError(409, "SFT_RESUME_SOURCE_UNAVAILABLE", "原训练缺少一致的预检及执行配置记录")
    automatic = snapshot.get("automatic_preflight")
    plan = automatic.get("plan") if isinstance(automatic, dict) else None
    datasets = plan.get("datasets") if isinstance(plan, dict) else None
    if (not isinstance(datasets, list) or len(datasets) != 1
            or automatic.get("plan_digest") != canonical_digest(plan)
            or plan.get("entrypoint") != str(PurePosixPath(snapshot["code_dir"]) / "scripts/train.py")
            or expectation.get("requested_entrypoint") is not None
            or expectation.get("training_stage") is not None):
        raise APIError(422, "SFT_RESUME_UNSUPPORTED", "当前仅支持使用默认 scripts/train.py 的单数据集 SFT 续训")
    return descriptor, run


def freeze_resume_source(session, node, data, parameters, request, *, frozen=False):
    """Resolve and recheck a resume request under existing admission locks."""
    if data.get("resume") is not True:
        if data.get("resume_source") is not None:
            raise APIError(409, "SFT_RESUME_SOURCE_MISMATCH", "续训来源与本次请求不一致，请重新预检")
        data.pop("resume", None)
        return
    if data["template_id"] != "openpi-sft" or data.get("entrypoint") is not None:
        raise APIError(422, "SFT_RESUME_UNSUPPORTED", "当前仅支持使用默认 scripts/train.py 的单数据集 SFT 续训")
    source, original = _source(session, node, data["inputs"].get("model"))
    require_resource_owner(request, original.owner_user_id)
    expectation = original.code_expectation
    definition = expectation.get("recipe_definition")
    original_choices = {
        "node_id": original.node_id, "profile_id": original.profile_id, "gpu_count": original.gpu_count,
        "recipe_revision_id": definition.get("id") if isinstance(definition, dict) else None,
        **{key: expectation.get(key) for key in (
            "recipe", "recipe_definition", "dataset_bindings", "auto_config", "normalization_method")},
    }
    if (any(canonical_digest(data.get(key)) != canonical_digest(value) for key, value in original_choices.items())
            or {key: value for key, value in data["inputs"].items() if key != "model"}
            != {key: value for key, value in original.inputs.items() if key != "model"}
            or canonical_digest({key: value for key, value in parameters.items() if key != "steps"})
            != canonical_digest({key: value for key, value in original.parameters.items() if key != "steps"})):
        raise APIError(409, "SFT_RESUME_CONFIG_MISMATCH", "续训须沿用原训练的机器、运行环境、配方、数据及参数，仅可调整目标总步数")
    if parameters["steps"] <= source["step"] + 1:
        raise APIError(422, "SFT_RESUME_STEPS_INVALID", "目标总步数必须大于检查点恢复后的训练步数")
    if frozen and data.get("resume_source") != source:
        raise APIError(409, "SFT_RESUME_SOURCE_MISMATCH", "续训检查点身份已变化，请重新预检")
    data["resume_source"] = source


def resume_preflight_error(session, request, snapshot, plan, checks):
    try:
        source = request.get("resume_source")
        resuming = request.get("resume") is True
        if resuming != (source is not None):
            raise ValueError("Resume mode and source differ")
        proof = validate_resume_evidence(plan, source)
        if not resuming:
            return None
        attempt = session.get(Attempt, source["attempt_id"])
        original_plan = (attempt.code_snapshot or {}).get("automatic_preflight") if attempt else None
        datasets = plan.get("datasets")
        if (isinstance(original_plan, dict) and proof["source_execution_plan_digest"] == original_plan.get("plan_digest")
                and any(check.get("name") == "resume_checkpoint" for check in checks)
                and isinstance(datasets, list) and len(datasets) == 1
                and plan.get("entrypoint") == str(PurePosixPath(snapshot["code_dir"]) / "scripts/train.py")):
            return None
    except (KeyError, TypeError, ValueError):
        pass
    return {"code": "SFT_RESUME_PLAN_MISMATCH", "message": "Agent 未确认本次续训检查点及原训练状态，请重新预检"}


def queued_resume_error(session, node, run):
    frozen = run.code_expectation.get("resume_source")
    if frozen is None:
        return None
    try:
        actual, original = _source(session, node, run.inputs.get("model"))
        if actual != frozen:
            raise APIError(409, "SFT_RESUME_SOURCE_MISMATCH", "续训检查点身份已变化，请重新预检")
        if original.owner_user_id != run.owner_user_id:
            actor = session.get(User, run.owner_user_id) if run.owner_user_id is not None else None
            if actor is None or actor.role != "admin":
                raise APIError(403, "RESOURCE_ACCESS_DENIED", "只能续训自己创建的任务")
    except APIError as exc:
        return {"code": exc.code, "message": exc.message}
    return None
