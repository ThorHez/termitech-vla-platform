"""Administrator audit filters and signed, filter-bound keyset pagination."""

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
from uuid import UUID

from sqlalchemy import and_, or_, select

from .auth_models import AuditEvent
from .common import APIError


UUID_TEXT = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
BASE64_TEXT = re.compile(r"[A-Za-z0-9_-]+\Z")
OUTCOMES = frozenset({"succeeded", "accepted", "denied", "failed"})


def _invalid_filter():
    return APIError(422, "INVALID_AUDIT_FILTER", "审计筛选条件无效")


def _timestamp(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise _invalid_filter()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except (ValueError, TypeError, OverflowError):
        raise _invalid_filter() from None


def _uuid(value):
    if not isinstance(value, str) or UUID_TEXT.fullmatch(value) is None:
        raise _invalid_filter()
    return str(UUID(value))


def _text(value, maximum, *, empty=False):
    if value is None:
        return None
    if (not isinstance(value, str) or len(value) > maximum
            or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)):
        raise _invalid_filter()
    value = value.strip()
    if not value:
        if empty:
            return None
        raise _invalid_filter()
    return value


def normalize_filters(*, limit=50, actor_id=None, action=None, outcome=None,
                      from_time=None, to_time=None, q=None):
    if (type(limit) is not int or not 1 <= limit <= 200
            or outcome is not None and (not isinstance(outcome, str) or outcome not in OUTCOMES)):
        raise _invalid_filter()
    actor_id = actor_id if actor_id is None or actor_id == "unattributed" else _uuid(actor_id)
    first = _timestamp(from_time) if from_time is not None else None
    last = _timestamp(to_time) if to_time is not None else None
    if first is not None and last is not None and first > last:
        raise _invalid_filter()
    return {"limit": limit, "actor_id": actor_id, "action": _text(action, 120), "outcome": outcome,
            "from_time": first, "to_time": last, "q": _text(q, 100, empty=True)}


def _fingerprint(filters):
    return hashlib.sha256(json.dumps(filters, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _key(settings):
    # Stable between controller workers/restarts; never expose enrollment data.
    return hmac.digest(settings.enrollment_token.encode(), b"vla-audit-cursor-v1", "sha256")


def _encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _decode(value):
    if not isinstance(value, str) or BASE64_TEXT.fullmatch(value) is None:
        raise ValueError
    decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    if _encode(decoded) != value:
        raise ValueError
    return decoded


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError
        value[key] = item
    return value


def encode_cursor(row, filters, settings):
    payload = json.dumps({"v": 1, "at": row.created_at, "id": row.id, "filters": _fingerprint(filters)},
                         sort_keys=True, separators=(",", ":")).encode()
    return _encode(payload) + "." + _encode(hmac.digest(_key(settings), payload, "sha256"))


def decode_cursor(cursor, filters, settings):
    try:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 1024:
            raise ValueError
        encoded, signature = cursor.split(".")
        payload, supplied = _decode(encoded), _decode(signature)
        if not hmac.compare_digest(hmac.digest(_key(settings), payload, "sha256"), supplied):
            raise ValueError
        value = json.loads(payload, object_pairs_hook=_object)
        if (not isinstance(value, dict) or set(value) != {"v", "at", "id", "filters"}
                or type(value["v"]) is not int or value["v"] != 1
                or value["filters"] != _fingerprint(filters)):
            raise ValueError
        if value["at"] != _timestamp(value["at"]) or value["id"] != _uuid(value["id"]):
            raise ValueError
        return value["at"], value["id"]
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError, APIError):
        raise APIError(422, "INVALID_AUDIT_CURSOR", "审计分页已失效，请重新查询") from None


def outcome_for(status):
    if status == 202:
        return "accepted"
    if status in {401, 403}:
        return "denied"
    return "failed" if status >= 400 else "succeeded"


def event_dict(row, *, include_outcome=True):
    from .auth import sanitize_audit_details
    value = {field: getattr(row, field) for field in (
        "id", "created_at", "actor_id", "actor_username", "action", "method", "path", "status", "remote_addr")}
    value["details"] = sanitize_audit_details(row.details)
    if include_outcome:
        value["outcome"] = outcome_for(row.status)
    return value


def query_events(session, settings, *, cursor=None, **parameters):
    filters = normalize_filters(**parameters)
    query = select(AuditEvent)
    if filters["actor_id"] == "unattributed":
        query = query.where(AuditEvent.actor_id.is_(None))
    elif filters["actor_id"] is not None:
        query = query.where(AuditEvent.actor_id == filters["actor_id"])
    if filters["action"] is not None:
        query = query.where(AuditEvent.action == filters["action"])
    if filters["outcome"] == "accepted":
        query = query.where(AuditEvent.status == 202)
    elif filters["outcome"] == "denied":
        query = query.where(AuditEvent.status.in_((401, 403)))
    elif filters["outcome"] == "failed":
        query = query.where(AuditEvent.status >= 400, AuditEvent.status.not_in((401, 403)))
    elif filters["outcome"] == "succeeded":
        query = query.where(AuditEvent.status < 400, AuditEvent.status != 202)
    if filters["from_time"] is not None:
        query = query.where(AuditEvent.created_at >= filters["from_time"])
    if filters["to_time"] is not None:
        query = query.where(AuditEvent.created_at <= filters["to_time"])
    if filters["q"] is not None:
        # Treat SQL wildcard characters as literal search text on both DBs.
        text = filters["q"].replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = "%" + text + "%"
        fields = [AuditEvent.action, AuditEvent.actor_username, AuditEvent.actor_id, AuditEvent.path]
        fields += [AuditEvent.details[key].as_string() for key in (
            "operation_label", "target_id", "target_name", "attempted_username")]
        query = query.where(or_(*(field.ilike(pattern, escape="\\") for field in fields)))
    if cursor is not None:
        stamp, identifier = decode_cursor(cursor, filters, settings)
        query = query.where(or_(AuditEvent.created_at < stamp,
            and_(AuditEvent.created_at == stamp, AuditEvent.id < identifier)))
    rows = list(session.scalars(query.order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc()).limit(filters["limit"] + 1)))
    more = len(rows) > filters["limit"]
    rows = rows[:filters["limit"]]
    return {"items": [event_dict(row) for row in rows], "has_more": more,
            "next_cursor": encode_cursor(rows[-1], filters, settings) if more else None}
