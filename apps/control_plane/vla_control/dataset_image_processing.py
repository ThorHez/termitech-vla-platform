"""Bind image choices to trusted machine-local metadata before registration."""

from copy import deepcopy

from vla_adapters.image_policy import validate_profile_image_processing

from .common import APIError
from .dataset_inspections import query_dict, validated_path
from .models import Command


def validate_image_choices(profile, features):
    try:
        return validate_profile_image_processing(profile, features)
    except ValueError as exc:
        code, _, message = str(exc).partition(": ")
        raise APIError(422, code, message or str(exc)) from None


def retain_legacy_image_augmentation(profile, previous_profile):
    """Preserve the legacy fallback when an editor only changes geometry."""
    previous = {(row["key"], row["kind"]): row.get("image_processing", {})
                for row in (previous_profile or {}).get("cameras", [])}
    for camera in profile["cameras"]:
        policy = camera.get("image_processing")
        legacy = previous.get((camera["key"], camera["kind"]))
        if policy is not None and isinstance(legacy, dict) and "augmentation" in legacy:
            policy.setdefault("augmentation", legacy["augmentation"])


def image_registration_evidence(profile, features, *, source, **identity):
    """Retain bounded source facts, without copying dataset contents or pixels."""
    validate_image_choices(profile, features)
    keys = {row["key"] for row in (profile or {}).get("cameras", [])}
    images = [deepcopy(row) for row in features if row.get("dtype") in {"image", "video"} or row["key"] in keys]
    return {"schema_version": 1, **identity, "source": deepcopy(source), "features": images}


def registration_evidence(session, node, path, profile, *, inspection_id=None, existing_metadata=None):
    """Use completed Agent evidence, never client-supplied dimensions.

    Unprofiled historical registrations retain the path-only API. A profiled
    registration must have real metadata, while old assets can use their
    previously saved trusted evidence when adding or editing choices.
    """
    path = validated_path(path, node.roots)
    if inspection_id is not None:
        command = session.get(Command, inspection_id)
        if (command is None or command.type != "DATASET_INSPECT" or command.node_id != node.id
                or command.payload.get("path") != path):
            raise APIError(422, "DATASET_IMAGE_INSPECTION_MISMATCH", "图像元信息读取记录与当前机器或数据集路径不一致，请重新读取")
        query = query_dict(command, node.roots)
        if query["status"] != "READY" or query["inspection"] is None:
            raise APIError(409, "DATASET_IMAGE_INSPECTION_REQUIRED", "请先完成当前数据集的元信息读取，再选择图像处理方式")
        inspection = query["inspection"]
        return image_registration_evidence(profile, inspection["features"], source=inspection["source"],
                                           inspection_id=inspection_id)
    evidence = (existing_metadata or {}).get("_image_registration")
    if isinstance(evidence, dict) and isinstance(evidence.get("features"), list):
        # The namespace is reserved at every client write boundary. Runtime
        # preflight still verifies metadata fingerprints against the files.
        validate_image_choices(profile, evidence["features"])
        return deepcopy(evidence)
    if profile is not None:
        raise APIError(409, "DATASET_IMAGE_INSPECTION_REQUIRED", "请先读取当前数据集元信息；非 224×224 图像必须明确选择处理方式")
    return None
