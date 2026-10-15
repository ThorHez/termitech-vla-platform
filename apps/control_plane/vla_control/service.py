"""State transitions shared by HTTP handlers and the scheduler."""

from datetime import datetime, timezone

from sqlalchemy import delete, select
from vla_adapters import get_templates, validate_parameters

from .common import APIError, INPUT_KINDS, TERMINAL, digest, now, require_path, snapshot_identity, uid, under_root, valid_snapshot
from .models import Allocation, Asset, Attempt, Command, Event, Node, Preview, Profile, Run
from .scheduler import enqueue


def get_required(session, model, identifier, label="Record"):
    value = session.get(model, identifier)
    if value is None:
        raise APIError(404, "NOT_FOUND", f"{label} not found")
    return value


def validate_request(session, data: dict, require_online=True) -> tuple[Node, Profile, dict]:
    node = get_required(session, Node, data["node_id"], "Node")
    if require_online and node.status != "ONLINE":
        raise APIError(409, "NODE_OFFLINE", "Node must be online before preflight or submission")
    profile = session.get(Profile, (node.id, data["profile_id"]))
    if profile is None or not profile.active:
        raise APIError(422, "PROFILE_NODE_MISMATCH", "Runtime profile does not belong to this node")
    template = next((t for t in get_templates() if t["id"] == data["template_id"]), None)
    if not template or template["status"] != "AVAILABLE":
        raise APIError(422, "TEMPLATE_UNAVAILABLE", "Template has not been made available")
    if profile.adapter != template["id"]:
        raise APIError(422, "PROFILE_TEMPLATE_MISMATCH", "Runtime profile does not support this template")
    if profile.execution_mode != "native":
        raise APIError(422, "EXECUTION_MODE_UNSUPPORTED", "This release supports registered native environments")
    if template["id"] != "smoke" and profile.metadata_json.get("validated") is not True:
        raise APIError(422, "PROFILE_NOT_VALIDATED", "Administrator must validate the local training recipe")
    if data["gpu_count"] < template.get("min_gpus", 1):
        raise APIError(422, "GPU_REQUIRED", "The selected template requires a GPU")
    if data["gpu_count"] > len(node.gpus):
        raise APIError(422, "GPU_CAPACITY_EXCEEDED", "Requested GPU count exceeds this node's reported devices")
    try:
        parameters = validate_parameters(template["id"], data["parameters"])
    except ValueError as exc:
        raise APIError(422, "INVALID_PARAMETERS", str(exc)) from exc
    if data["gpu_count"] and parameters["batch_size"] % data["gpu_count"]:
        raise APIError(422, "INVALID_BATCH_SIZE", "Batch size must be divisible by GPU count")
    if set(data["inputs"]) - INPUT_KINDS.keys():
        raise APIError(422, "UNKNOWN_INPUT_ROLE", "Allowed input roles are dataset, model and norm")
    if set(template["required_inputs"]) - data["inputs"].keys():
        raise APIError(422, "INPUT_REQUIRED", "A required local training asset is missing")
    for role, asset_id in data["inputs"].items():
        asset = get_required(session, Asset, asset_id, "Asset")
        if asset.node_id != node.id:
            raise APIError(422, "ASSET_NODE_MISMATCH", "All assets and the runtime must be on the selected node")
        if asset.kind != INPUT_KINDS[role]:
            raise APIError(422, "ASSET_KIND_MISMATCH", "Asset type does not match the input role")
        if asset.state not in {"REGISTERED", "READY"}:
            raise APIError(409, "INPUT_UNAVAILABLE", "An input asset is archived or unavailable")
        require_path(asset.path, node.roots)
    return node, profile, parameters


def sync_profiles(session, node: Node, entries: list[dict]) -> dict:
    """Update node-owned environments without mutating a live run's runtime."""
    if len({entry["id"] for entry in entries}) != len(entries):
        return {"status": "REJECTED", "code": "DUPLICATE_PROFILE", "message": "Profile IDs must be unique"}
    for entry in entries:
        try:
            require_path(entry["code_dir"], node.roots)
            require_path(entry["python"])
        except APIError as exc:
            return {"status": "REJECTED", "code": exc.code, "message": exc.message}
    existing = {p.id: p for p in session.scalars(select(Profile).where(Profile.node_id == node.id).with_for_update())}
    incoming = {entry["id"]: entry for entry in entries}
    changed = set()
    for profile_id in existing.keys() | incoming.keys():
        current, proposed = existing.get(profile_id), incoming.get(profile_id)
        if current is None:
            changed.add(profile_id)
        elif proposed is None:
            if current.active:
                changed.add(profile_id)
        elif not current.active or any(getattr(current, "metadata_json" if key == "metadata" else key) != value for key, value in proposed.items()):
            changed.add(profile_id)
    used = set(session.scalars(select(Run.profile_id).where(Run.node_id == node.id, Run.state.not_in(TERMINAL), Run.profile_id.in_(changed)))) if changed else set()
    applied = changed - used
    for profile_id in applied:
        current, proposed = existing.get(profile_id), incoming.get(profile_id)
        if proposed is None:
            current.active = False  # Historical runs still reference this row.
        else:
            values = {("metadata_json" if key == "metadata" else key): value for key, value in proposed.items()}
            if current is None:
                session.add(Profile(node_id=node.id, active=True, **values))
            else:
                for key, value in values.items():
                    setattr(current, key, value)
                current.active = True
    if applied:
        for preview in session.scalars(select(Preview).where(Preview.node_id == node.id, Preview.status.in_(["PENDING", "READY"])).with_for_update()):
            if preview.request["profile_id"] in applied:
                preview.status = "FAILED"
                preview.error = {"code": "PROFILE_CHANGED_SINCE_PREFLIGHT", "message": "Runtime configuration changed; run preflight again"}
    if used:
        return {"status": "BLOCKED", "code": "PROFILE_IN_USE", "message": "Active runs retain their registered runtime configuration",
                "blocked_profile_ids": sorted(used), "updated_profile_ids": sorted(applied)}
    return {"status": "APPLIED", "updated_profile_ids": sorted(applied)}


def apply_ack(session, command: Command, status: str, result: dict) -> None:
    if command.status != "PENDING":
        if command.result == {"terminal_event_received": True}:
            return
        if command.status != status or digest(command.result) != digest(result):
            raise APIError(409, "ACK_CONFLICT", "This command already has a different acknowledgement")
        return
    command.status, command.result, command.acknowledged_at = status, result, now()
    if command.type != "PREFLIGHT":
        # Acknowledgement proves delivery, never process exit. Reservations are
        # released exclusively by a terminal execution event.
        return
    preview = session.scalar(select(Preview).where(Preview.id == command.payload["preview_id"]).with_for_update())
    if preview.status != "PENDING":
        return
    if preview.expires_at < now():
        preview.status = "EXPIRED"
        preview.error = {"code": "PREFLIGHT_EXPIRED", "message": "Preflight expired"}
        return
    checks = result.get("checks", [])
    snapshot = result.get("code_snapshot")
    error = result.get("error")
    if not isinstance(checks, list) or not checks or any(not isinstance(c, dict) or c.get("ok") is not True for c in checks):
        error = error or {"code": "PREFLIGHT_FAILED", "message": "Node validation failed or returned no checks"}
    if not valid_snapshot(snapshot):
        error = error or {"code": "CODE_VERSION_UNAVAILABLE", "message": "Complete actual code version could not be established"}
    if status == "REJECTED":
        error = error or {"code": "PREFLIGHT_REJECTED", "message": "Agent rejected the preflight command"}
    preview.checks = checks if isinstance(checks, list) else []
    preview.code_snapshot = snapshot if isinstance(snapshot, dict) else None
    preview.error = error
    preview.status = "FAILED" if error else "READY"
    if not error:
        for asset_id in preview.request["inputs"].values():
            asset = session.get(Asset, asset_id)
            if asset.state == "REGISTERED":
                asset.state = "READY"


def canonical_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError) as exc:
        raise APIError(422, "INVALID_TIMESTAMP", "Event timestamps must include a timezone") from exc


def event_failure(session, attempt, run, code: str, message: str) -> None:
    run.error = attempt.error = {"code": code, "message": message}
    if run.state != "CANCELLING":
        enqueue(session, run.node_id, "CANCEL", {"attempt_id": attempt.id, "execution_token": attempt.execution_token})
    run.state = attempt.state = "CANCELLING"


def reduce_event(session, node: Node, attempt: Attempt, event: Event) -> None:
    run = session.scalar(select(Run).where(Run.id == attempt.run_id).with_for_update())
    data = event.data
    if event.type == "LOG":
        if not isinstance(data.get("text"), str) or len(data["text"]) > 100000:
            raise APIError(422, "INVALID_EVENT", "Log events need bounded text")
        return
    if event.seq < attempt.last_event_seq:
        return
    if event.type == "METRIC":
        attempt.latest_metrics = {**attempt.latest_metrics, **data, "timestamp": event.timestamp}
        return
    if event.type == "RESOURCE":
        attempt.resources = {**data, "timestamp": event.timestamp}
        return
    if attempt.state in TERMINAL:
        if event.type in TERMINAL and event.type != attempt.state:
            raise APIError(409, "TERMINAL_STATE_CONFLICT", "Attempt already has a different terminal state")
        return
    run.updated_at = now()
    if event.type == "STARTED":
        snapshot = data.get("code_snapshot")
        if attempt.code_snapshot is not None and snapshot != attempt.code_snapshot:
            raise APIError(409, "CODE_SNAPSHOT_IMMUTABLE", "Actual execution code snapshot is immutable")
        attempt.code_snapshot = snapshot
        attempt.started_at = attempt.started_at or event.timestamp
        pid = data.get("pid")
        attempt.pid = pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None
        attempt.output_dir = data.get("output_dir")
        if not valid_snapshot(snapshot) or snapshot_identity(snapshot) != snapshot_identity(run.code_expectation):
            event_failure(session, attempt, run, "CODE_CHANGED_SINCE_PREFLIGHT", "Reported execution code differs from the frozen preflight")
        elif not isinstance(attempt.output_dir, str) or not under_root(attempt.output_dir, [node.output_root]):
            event_failure(session, attempt, run, "OUTPUT_PATH_INVALID", "Runner output directory is outside the registered output root")
        elif run.state == "CANCELLING" or run.previous_state == "CANCELLING":
            run.state = attempt.state = "CANCELLING"
        else:
            run.state = attempt.state = "RUNNING"
        return
    if event.type not in TERMINAL:
        return
    exit_code = data.get("exit_code")
    no_process_started = (data.get("process_started") is False and attempt.pid is None and attempt.started_at is None)
    if (not isinstance(exit_code, int) or isinstance(exit_code, bool)) and not (exit_code is None and no_process_started and event.type in {"FAILED", "CANCELLED"}):
        raise APIError(422, "EXIT_NOT_CONFIRMED", "A terminal event must confirm the process exit code")
    if attempt.code_snapshot is None and isinstance(data.get("code_snapshot"), dict):
        attempt.code_snapshot = data["code_snapshot"]
    attempt.code_changed_during_run = data.get("code_changed_during_run") if isinstance(data.get("code_changed_during_run"), bool) else None
    final_state = event.type
    failure = data.get("error") if event.type == "FAILED" else None
    output_rows = []
    if event.type == "SUCCEEDED":
        outputs = data.get("outputs", [])
        if exit_code != 0 or not isinstance(outputs, list):
            failure = {"code": "OUTPUT_INCOMPLETE", "message": "Successful execution needs exit code zero and an output manifest"}
        elif run.template_id != "smoke" and not any(isinstance(o, dict) and o.get("kind") == "MODEL" for o in outputs):
            failure = {"code": "OUTPUT_INCOMPLETE", "message": "Training completed without a model artifact"}
        else:
            for output in outputs:
                if (not isinstance(output, dict) or output.get("kind") not in set(INPUT_KINDS.values())
                    or not isinstance(output.get("path"), str) or not under_root(output["path"], [node.output_root])
                    or not isinstance(output.get("name"), str) or not output["name"]
                    or not isinstance(output.get("metadata", {}), dict)):
                    failure = {"code": "OUTPUT_INVALID", "message": "Output manifest has an invalid kind or local path"}
                    break
                output_rows.append(Asset(id=uid(), node_id=node.id, name=output["name"][:200], kind=output["kind"],
                    path=output["path"], version=str(output.get("version", "1"))[:200], state="READY",
                    metadata_json=output.get("metadata", {}), created_at=now(), origin_attempt_id=attempt.id))
        if attempt.code_snapshot is None:
            failure = {"code": "CODE_VERSION_UNAVAILABLE", "message": "No execution code snapshot was recorded"}
    if attempt.error:
        failure = attempt.error
    if failure or event.type == "FAILED":
        final_state = "FAILED"
        if not isinstance(failure, dict):
            failure = {"code": "TRAINING_FAILED", "message": str(failure or "Runner reported a failed execution")}
    if final_state == "SUCCEEDED":
        session.add_all(output_rows)
    attempt.state = run.state = final_state
    attempt.exit_code = exit_code
    attempt.error = run.error = failure
    attempt.finished_at = run.finished_at = event.timestamp
    run.queue_reason = None
    run.previous_state = attempt.previous_state = None
    session.execute(delete(Allocation).where(Allocation.attempt_id == attempt.id))
    # Pending deliveries are no longer useful after confirmed exit. Keep the
    # command history, but prevent a late START from being redelivered forever.
    for command in session.scalars(select(Command).where(Command.node_id == node.id, Command.status == "PENDING")):
        if command.payload.get("attempt_id") == attempt.id:
            command.status = "APPLIED"
            command.result = {"terminal_event_received": True}
            command.acknowledged_at = now()


def ingest_events(session, node: Node, entries: list[dict]) -> int:
    accepted = 0
    for item in sorted(entries, key=lambda e: (e["attempt_id"], e["seq"])):
        attempt = session.scalar(select(Attempt).where(Attempt.id == item["attempt_id"]).with_for_update())
        if attempt is None or attempt.node_id != node.id:
            raise APIError(403, "NODE_OWNERSHIP_MISMATCH", "Attempt is not assigned to this node")
        item = {**item, "timestamp": canonical_timestamp(item["timestamp"])}
        event_digest = digest(item)
        existing = session.get(Event, (item["attempt_id"], item["seq"]))
        if existing:
            if existing.digest != event_digest:
                raise APIError(409, "EVENT_SEQUENCE_CONFLICT", "Sequence number was reused with different event data")
            accepted += 1
            continue
        event = Event(**item, digest=event_digest)
        session.add(event)
        reduce_event(session, node, attempt, event)
        attempt.last_event_seq = max(attempt.last_event_seq, item["seq"])
        session.flush()
        accepted += 1
    return accepted
