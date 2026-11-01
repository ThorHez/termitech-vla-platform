"""Lossless, metric-only history independent of the bounded live log feed."""

import math

from fastapi import Query
from sqlalchemy import func, select

from .common import APIError
from .models import Attempt, Event
from .run_records import require_visible_run


def _finite_json(value):
    """Keep invalid numeric observations as explicit gaps in valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return value


def metric_page(session, run_id, *, attempt_id=None, after_seq=0, limit=1000):
    require_visible_run(session, run_id)
    attempts = select(Attempt).where(Attempt.run_id == run_id)
    if attempt_id is not None:
        attempt = session.scalar(attempts.where(Attempt.id == attempt_id))
        if attempt is None:
            raise APIError(404, "NOT_FOUND", "Attempt not found for this run")
    else:
        attempt = session.scalar(attempts.order_by(Attempt.number.desc()).limit(1))
    if attempt is None:
        return {"attempt_id": None, "items": [], "next_seq": after_seq,
                "has_more": False, "total_count": 0}

    filters = (Event.attempt_id == attempt.id, Event.type == "METRIC")
    total_count = session.scalar(select(func.count()).select_from(Event).where(*filters))
    rows = list(session.scalars(select(Event).where(*filters, Event.seq > after_seq)
                                .order_by(Event.seq).limit(limit + 1)))
    items = rows[:limit]
    return {
        "attempt_id": attempt.id,
        "items": [{"attempt_id": event.attempt_id, "seq": event.seq, "type": event.type,
                   "timestamp": event.timestamp, "data": _finite_json(event.data)} for event in items],
        "next_seq": items[-1].seq if items else after_seq,
        "has_more": len(rows) > limit,
        "total_count": total_count,
    }


def register_routes(app, admin, db):
    @app.get("/api/v1/runs/{run_id}/metrics", dependencies=admin)
    def run_metrics(run_id: str, attempt_id: str | None = Query(default=None, min_length=1, max_length=36),
                    after_seq: int = Query(default=0, ge=0, le=9223372036854775807),
                    limit: int = Query(default=1000, ge=1, le=1000)):
        with db.read() as session:
            return metric_page(session, run_id, attempt_id=attempt_id, after_seq=after_seq, limit=limit)
