"""Verify ReCap annotation model provenance from the platform catalog."""

from sqlalchemy import func, or_, select
from vla_adapters.recap import STAGE_ANNOTATION, STAGE_PROGRESS, recap_model_role
from vla_adapters.recap_checkpoint import RECAP_TORCH_FORMAT, recap_torch_artifact_valid

from .common import under_root
from .model_formats import PYTORCH_FORMAT, normalize_model_format
from .models import Attempt, Node, Run


def is_external_stage_annotation_checkpoint(asset, annotation_stage):
    """Allow registered PyTorch stage scorers without asserting training provenance."""
    if (annotation_stage != STAGE_ANNOTATION or asset is None or asset.kind != "MODEL"
            or asset.origin_attempt_id is not None or asset.state not in {"REGISTERED", "READY"}):
        return False
    metadata = asset.metadata_json
    return (isinstance(metadata, dict) and metadata.get("availability") != "removed"
            and normalize_model_format(metadata.get("format")) == PYTORCH_FORMAT
            and metadata.get("model_role") in (None, recap_model_role(STAGE_PROGRESS)))


def stage_annotation_sources(session, assets, *, scorer_stage=STAGE_PROGRESS):
    """Return eligible checkpoint sources, without loading full run snapshots."""
    candidates = []
    for asset in assets:
        metadata = asset.metadata_json
        if (asset.kind == "MODEL" and asset.state == "READY" and asset.origin_attempt_id
                and isinstance(metadata, dict) and metadata.get("availability") != "removed"
                and metadata.get("format") == RECAP_TORCH_FORMAT
                and metadata.get("model_role") == recap_model_role(scorer_stage)
                and type(metadata.get("step")) is int and 0 <= metadata["step"] < 2**53):
            candidates.append(asset)
    if not candidates:
        return {}

    observed_stage = Attempt.code_snapshot["training_stage"].as_string()
    recipe = func.coalesce(Run.code_expectation["automatic_preflight"]["plan"]["recipe"].as_string(),
                           Run.code_expectation["recipe"].as_string())
    query = select(Attempt.id, Attempt.node_id, Attempt.output_dir, Run.id, Run.name, recipe, Node.roots).select_from(
        Attempt).join(Run, Run.id == Attempt.run_id).join(Node, Node.id == Attempt.node_id).where(
        Attempt.id.in_({asset.origin_attempt_id for asset in candidates}),
        Attempt.state == "SUCCEEDED", Attempt.finished_at.is_not(None), Attempt.exit_code == 0,
        or_(observed_stage.is_(None), observed_stage == scorer_stage),
        Run.node_id == Attempt.node_id, Run.template_id == "recap", Run.state == "SUCCEEDED",
        Run.deleted_at.is_(None), Run.code_expectation["training_stage"].as_string() == scorer_stage,
    )
    origins = {row[0]: row for row in session.execute(query)}
    sources = {}
    for asset in candidates:
        origin = origins.get(asset.origin_attempt_id)
        if origin is None:
            continue
        attempt_id, node_id, output_dir, run_id, run_name, recipe_name, roots = origin
        descriptor = {"kind": asset.kind, "path": asset.path, "version": asset.version, "metadata": asset.metadata_json}
        if (asset.node_id == node_id and under_root(asset.path, roots)
                and recap_torch_artifact_valid(descriptor, output_dir, recipe_name, scorer_stage)):
            sources[asset.id] = {"run_id": run_id, "run_name": run_name, "attempt_id": attempt_id,
                                 "step": asset.metadata_json["step"]}
    return sources
