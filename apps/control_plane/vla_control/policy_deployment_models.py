"""Durable local inference deployments; weights remain on the owning machine."""

from sqlalchemy import JSON, ForeignKey, ForeignKeyConstraint, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


RESOURCE_STATES = frozenset({"PENDING", "STARTING", "RUNNING", "STOPPING", "UNKNOWN"})


class PolicyDeployment(Base):
    __tablename__ = "policy_deployments"
    __table_args__ = (
        ForeignKeyConstraint(["node_id", "profile_id"], ["profiles.node_id", "profiles.id"], ondelete="RESTRICT"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    profile_id: Mapped[str] = mapped_column(String(200))
    checkpoint_path: Mapped[str] = mapped_column(Text)
    config_name: Mapped[str] = mapped_column(String(255))
    inference_backend: Mapped[str] = mapped_column(String(16), default="auto", server_default="auto")
    resolved_backend: Mapped[str | None] = mapped_column(String(16), nullable=True)
    port: Mapped[int] = mapped_column(Integer)
    gpu_uuid: Mapped[str | None] = mapped_column(String(200), nullable=True)
    gpu_mode: Mapped[str] = mapped_column(String(16), default="exclusive", server_default="exclusive")
    gpu_memory_mb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    default_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="RESTRICT"), nullable=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    logs: Mapped[str] = mapped_column(Text, default="")
    checks: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    test_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    updated_at: Mapped[str] = mapped_column(String(40))
    observed_command_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
