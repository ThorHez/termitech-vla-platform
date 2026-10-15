"""ORM models; source files and model weights never enter this database."""

from sqlalchemy import Boolean, JSON, ForeignKey, ForeignKeyConstraint, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Node(Base):
    __tablename__ = "nodes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    registration_key: Mapped[str | None] = mapped_column(String(64), unique=True)
    registration_digest: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    hostname: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="REGISTERED")
    last_seen: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[str] = mapped_column(String(40))
    roots: Mapped[list] = mapped_column(JSON)
    output_root: Mapped[str] = mapped_column(Text)
    gpus: Mapped[list] = mapped_column(JSON, default=list)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)


class Profile(Base):
    __tablename__ = "profiles"
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), primary_key=True)
    id: Mapped[str] = mapped_column(String(200), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    code_dir: Mapped[str] = mapped_column(Text)
    python: Mapped[str] = mapped_column(Text)
    adapter: Mapped[str] = mapped_column(String(100))
    recipe: Mapped[str | None] = mapped_column(String(255))
    execution_mode: Mapped[str] = mapped_column(String(32), default="native")
    docker_image: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Asset(Base):
    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("id", "node_id"),
        ForeignKeyConstraint(["origin_attempt_id", "node_id"], ["attempts.id", "attempts.node_id"], ondelete="RESTRICT"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(20))
    path: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(200))
    state: Mapped[str] = mapped_column(String(32), default="REGISTERED")
    metadata_json: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[str] = mapped_column(String(40))
    origin_attempt_id: Mapped[str | None] = mapped_column(String(36), index=True)


class Preview(Base):
    __tablename__ = "previews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    request: Mapped[dict] = mapped_column(JSON)
    checks: Mapped[list] = mapped_column(JSON, default=list)
    code_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40))
    expires_at: Mapped[str] = mapped_column(String(40))
    run_id: Mapped[str | None] = mapped_column(String(36), unique=True)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint("id", "node_id"),
        ForeignKeyConstraint(["node_id", "profile_id"], ["profiles.node_id", "profiles.id"], ondelete="RESTRICT"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    profile_id: Mapped[str] = mapped_column(String(200))
    template_id: Mapped[str] = mapped_column(String(100))
    state: Mapped[str] = mapped_column(String(32), default="QUEUED", index=True)
    previous_state: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    updated_at: Mapped[str] = mapped_column(String(40))
    finished_at: Mapped[str | None] = mapped_column(String(40))
    inputs: Mapped[dict] = mapped_column(JSON)
    parameters: Mapped[dict] = mapped_column(JSON)
    gpu_count: Mapped[int] = mapped_column(Integer)
    code_expectation: Mapped[dict] = mapped_column(JSON)
    queue_reason: Mapped[str | None] = mapped_column(Text)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class RunInput(Base):
    __tablename__ = "run_inputs"
    __table_args__ = (
        ForeignKeyConstraint(["run_id", "node_id"], ["runs.id", "runs.node_id"], ondelete="RESTRICT"),
        ForeignKeyConstraint(["asset_id", "node_id"], ["assets.id", "assets.node_id"], ondelete="RESTRICT"),
    )
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    role: Mapped[str] = mapped_column(String(32), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(36))
    asset_id: Mapped[str] = mapped_column(String(36))


class Attempt(Base):
    __tablename__ = "attempts"
    __table_args__ = (
        ForeignKeyConstraint(["run_id", "node_id"], ["runs.id", "runs.node_id"], ondelete="RESTRICT"),
        UniqueConstraint("run_id", "number"),
        UniqueConstraint("id", "node_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    node_id: Mapped[str] = mapped_column(String(36), index=True)
    number: Mapped[int] = mapped_column(Integer, default=1)
    execution_token: Mapped[str] = mapped_column(String(64), unique=True)
    state: Mapped[str] = mapped_column(String(32), default="STARTING")
    previous_state: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(40))
    started_at: Mapped[str | None] = mapped_column(String(40))
    finished_at: Mapped[str | None] = mapped_column(String(40))
    pid: Mapped[int | None] = mapped_column(Integer)
    code_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    gpu_uuids: Mapped[list] = mapped_column(JSON, default=list)
    output_dir: Mapped[str | None] = mapped_column(Text)
    latest_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    resources: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    last_event_seq: Mapped[int] = mapped_column(Integer, default=0)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    code_changed_during_run: Mapped[bool | None] = mapped_column(Boolean)


class Allocation(Base):
    __tablename__ = "gpu_allocations"
    __table_args__ = (
        ForeignKeyConstraint(["attempt_id", "node_id"], ["attempts.id", "attempts.node_id"], ondelete="RESTRICT"),
    )
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), primary_key=True)
    gpu_uuid: Mapped[str] = mapped_column(String(200), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[str] = mapped_column(String(40))


class Command(Base):
    __tablename__ = "commands"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    acknowledged_at: Mapped[str | None] = mapped_column(String(40))


class Event(Base):
    __tablename__ = "events"
    attempt_id: Mapped[str] = mapped_column(ForeignKey("attempts.id", ondelete="RESTRICT"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[str] = mapped_column(String(40))
    data: Mapped[dict] = mapped_column(JSON)
    digest: Mapped[str] = mapped_column(String(64))


class Idempotency(Base):
    __tablename__ = "idempotency_keys"
    key: Mapped[str] = mapped_column(String(200), primary_key=True)
    digest: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="RESTRICT"))
    created_at: Mapped[str] = mapped_column(String(40))
