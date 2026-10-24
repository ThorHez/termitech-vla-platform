"""Persistent storage and transactional write boundaries for SQLite/PostgreSQL."""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str):
        parsed = make_url(url)
        options = {"pool_pre_ping": True}
        if parsed.get_backend_name() == "sqlite":
            options["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if parsed.database in (None, "", ":memory:"):
                options["poolclass"] = StaticPool
            else:
                Path(parsed.database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, **options)
        if self.engine.dialect.name == "sqlite":
            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=30000")
                if parsed.database not in (None, "", ":memory:"):
                    connection.execute("PRAGMA journal_mode=WAL")
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        from . import models  # noqa: F401
        from . import remote_models  # noqa: F401
        from . import recipe_models  # noqa: F401
        from . import auth_models  # noqa: F401
        Base.metadata.create_all(self.engine)
        # One stable row serializes administrator invariants across processes.
        with self.engine.begin() as connection:
            if self.engine.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            connection.exec_driver_sql("INSERT INTO auth_guard (id) VALUES (1) ON CONFLICT (id) DO NOTHING")
            # Authentication-only additive upgrade. Historical cookies cannot
            # prove human activity under the new policy, so revoke those
            # sessions while preserving users, Agents and all business data.
            if self.engine.dialect.name == "postgresql":
                connection.exec_driver_sql("SELECT id FROM auth_guard WHERE id = 1 FOR UPDATE")
            columns = {column["name"] for column in inspect(connection).get_columns("user_sessions")}
            missing = {"last_activity_at", "reauthenticated_at"} - columns
            for column in sorted(missing):
                connection.exec_driver_sql(f"ALTER TABLE user_sessions ADD COLUMN {column} VARCHAR(40)")
            if missing:
                connection.exec_driver_sql("DELETE FROM user_sessions")
            # Keep custom labels independent of Agent registration and upgrade
            # fields. The same guard serializes this additive migration without
            # revoking current sessions or changing historical business rows.
            node_columns = {column["name"] for column in inspect(connection).get_columns("nodes")}
            if "display_name" not in node_columns:
                connection.exec_driver_sql("ALTER TABLE nodes ADD COLUMN display_name VARCHAR(200)")

    @contextmanager
    def read(self):
        with self.sessions() as session:
            yield session

    @contextmanager
    def write(self):
        with self.sessions() as session:
            try:
                # Serialize SQLite writers across processes. PostgreSQL uses row
                # locks on nodes and runs at the specific mutation sites.
                if self.engine.dialect.name == "sqlite":
                    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    def close(self) -> None:
        self.engine.dispose()
