"""Durable checkpoint transfer receipts; large weights stay outside the controller."""

from sqlalchemy import JSON, BigInteger, Float, ForeignKey, ForeignKeyConstraint, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


TRANSFER_ACTIVE = frozenset({"QUEUED", "PREPARING", "EXPORTING", "TRANSFERRING", "VERIFYING"})
TRANSFER_RETRYABLE = frozenset({"FAILED", "CANCELLED", "INTERRUPTED"})
TRANSFER_TERMINAL = frozenset({"SUCCEEDED", *TRANSFER_RETRYABLE})


class CheckpointTransfer(Base):
    __tablename__ = "checkpoint_transfers"
    __table_args__ = (ForeignKeyConstraint(["target_node_id", "target_profile_id"],
        ["profiles.node_id", "profiles.id"], ondelete="RESTRICT"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    source_node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    source_run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="RESTRICT"), index=True)
    source_checkpoint_path: Mapped[str] = mapped_column(Text)
    target_node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    target_profile_id: Mapped[str] = mapped_column(String(200))
    target_directory: Mapped[str] = mapped_column(Text)
    target_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    bytes_total: Mapped[int] = mapped_column(BigInteger, default=0)
    bytes_transferred: Mapped[int] = mapped_column(BigInteger, default=0)
    files_total: Mapped[int] = mapped_column(Integer, default=0)
    # -1 is a persisted unknown count for rsync versions without per-file
    # aggregate progress. Keep the original NOT NULL column upgrade compatible;
    # the API serializes this sentinel as JSON null.
    files_transferred: Mapped[int] = mapped_column(Integer, default=0)
    speed_bytes_per_second: Mapped[float] = mapped_column(Float, default=0)
    progress_pct: Mapped[float] = mapped_column(Float, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    logs: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    portable_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    finished_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    binding_snapshot: Mapped[dict] = mapped_column(JSON)
    portable_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    manifest: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class PolicyDeploymentTransfer(Base):
    __tablename__ = "policy_deployment_transfers"
    deployment_id: Mapped[str] = mapped_column(ForeignKey("policy_deployments.id", ondelete="RESTRICT"), primary_key=True)
    transfer_id: Mapped[str] = mapped_column(ForeignKey("checkpoint_transfers.id", ondelete="RESTRICT"), index=True)
