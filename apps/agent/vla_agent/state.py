from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class State:
    """Agent and detached runners share a WAL-backed, transactional outbox."""

    def __init__(self, state_dir: str):
        directory = Path(state_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "agent.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS credentials (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL,
                    runner_identity TEXT, child_identity TEXT, output_dir TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    last_seq INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (attempt_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL,
                    timestamp TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(attempt_id,seq));
            """)
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def credentials(self) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT value FROM credentials WHERE id=1").fetchone()
            return json.loads(row[0]) if row else None

    def save_credentials(self, value: dict):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO credentials VALUES(1,?)", (json.dumps(value),))

    def command(self, command_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT status,result FROM commands WHERE id=?", (command_id,)).fetchone()
            return {"status": row[0], "result": json.loads(row[1])} if row else None

    def save_command(self, command_id: str, status: str, result: dict):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO commands VALUES(?,?,?)", (command_id, status, json.dumps(result)))

    def create_attempt(self, payload: dict, output_dir: str) -> bool:
        with self.connect() as db:
            cursor = db.execute("INSERT OR IGNORE INTO attempts(id,payload,state,output_dir,updated_at) VALUES(?,?,?,?,?)",
                                (payload["attempt_id"], json.dumps(payload), "STARTING", output_dir, utc_now()))
            if not cursor.rowcount:
                existing = json.loads(db.execute("SELECT payload FROM attempts WHERE id=?", (payload["attempt_id"],)).fetchone()[0])
                if existing != payload:
                    raise ValueError("Attempt identity reused with different payload")
            return bool(cursor.rowcount)

    @staticmethod
    def _decode(row) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("payload", "runner_identity", "child_identity"):
            result[key] = json.loads(result[key]) if result[key] else None
        return result

    def attempt(self, attempt_id: str) -> dict | None:
        with self.connect() as db:
            return self._decode(db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone())

    def active_attempts(self) -> list[dict]:
        with self.connect() as db:
            return [self._decode(row) for row in db.execute("SELECT * FROM attempts WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')")]

    def set_identity(self, attempt_id: str, kind: str, identity: dict):
        if kind not in {"runner", "child"}:
            raise ValueError("Unknown process identity type")
        with self.connect() as db:
            db.execute(f"UPDATE attempts SET {kind}_identity=?,updated_at=? WHERE id=?", (json.dumps(identity), utc_now(), attempt_id))

    def request_cancel(self, attempt_id: str, execution_token: str):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if not row or json.loads(row[0])["execution_token"] != execution_token:
                raise ValueError("Unknown attempt or execution token mismatch")
            db.execute("UPDATE attempts SET cancel_requested=1,updated_at=? WHERE id=?", (utc_now(), attempt_id))

    def emit(self, attempt_id: str, event_type: str, data: dict) -> int:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state,last_seq FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if not row:
                raise ValueError("Event for unknown attempt")
            if row[0] in TERMINAL_STATES:
                return row[1]
            seq = row[1] + 1
            timestamp = utc_now()
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (attempt_id, seq, event_type, timestamp, json.dumps(data)))
            next_state = event_type if event_type in TERMINAL_STATES else "RUNNING" if event_type == "STARTED" else row[0]
            db.execute("UPDATE attempts SET state=?,last_seq=?,updated_at=? WHERE id=?", (next_state, seq, timestamp, attempt_id))
            return seq

    def pending_events(self, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            return [{**dict(row), "data": json.loads(row["data"])} for row in db.execute("SELECT * FROM events ORDER BY attempt_id,seq LIMIT ?", (limit,))]

    def acknowledge_events(self, events: list[dict]):
        with self.connect() as db:
            db.executemany("DELETE FROM events WHERE attempt_id=? AND seq=?", [(e["attempt_id"], e["seq"]) for e in events])
