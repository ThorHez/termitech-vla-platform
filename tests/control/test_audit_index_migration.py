"""Audit history indexes upgrade existing tables without changing any record."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from threading import Barrier

from sqlalchemy import inspect

from vla_control.db import Database

from .conftest import ADMIN


INDEXES = {
    "ix_audit_events_created_id": ["created_at", "id"],
    "ix_audit_events_actor_created_id": ["actor_id", "created_at", "id"],
}


def fingerprint(database):
    # Hash fixture cells so even a failing preservation assertion cannot dump
    # password/session fields into a test report.
    with database.engine.connect() as connection:
        return {table: hashlib.sha256(json.dumps([tuple(row) for row in connection.exec_driver_sql(
            "SELECT * FROM " + table + " ORDER BY id")], default=str).encode()).hexdigest()
            for table in ("audit_events", "users", "user_sessions", "assets", "nodes", "profiles")}


def test_audit_composite_indexes_are_additive_concurrent_and_idempotent(platform):
    node = platform.register()
    response = platform.client.post("/api/v1/assets", headers=ADMIN, json={
        "name": "historical dataset", "kind": "DATASET", "node_id": node["node_id"], "path": "/machine/history"})
    assert response.status_code == 201
    password = "isolated-audit-index-migration-password"
    platform.app.state.auth.create_user("migration-admin", password, "admin")
    platform.client.base_url = "https://testserver"
    assert platform.client.post("/api/v1/auth/login", json={"username": "migration-admin", "password": password}).status_code == 200
    database = platform.app.state.db
    with database.engine.begin() as connection:
        for name in INDEXES:
            connection.exec_driver_sql("DROP INDEX " + name)
    before = fingerprint(database)
    barrier = Barrier(2)

    def migrate(_):
        independent = Database(platform.settings.database_url)
        try:
            barrier.wait(timeout=5)
            independent.initialize()
        finally:
            independent.close()

    with ThreadPoolExecutor(max_workers=2) as threads:
        list(threads.map(migrate, range(2)))
    database.initialize()
    assert fingerprint(database) == before
    indexes = {index["name"]: index["column_names"] for index in inspect(database.engine).get_indexes("audit_events")}
    assert all(indexes[name] == columns for name, columns in INDEXES.items())
    assert platform.client.get("/api/v1/auth/me").status_code == 200
    assert fingerprint(database) == before
