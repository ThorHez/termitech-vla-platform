"""Workflow ownership, independent from GPU service executions and transfer IO."""

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class DeploymentRecord(Base):
    __tablename__ = "deployment_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    merged_into_id: Mapped[str | None] = mapped_column(ForeignKey("deployment_records.id", ondelete="RESTRICT"), nullable=True)


class DeploymentRecordRuntime(Base):
    __tablename__ = "deployment_record_runtimes"
    deployment_id: Mapped[str] = mapped_column(ForeignKey("policy_deployments.id", ondelete="RESTRICT"), primary_key=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("deployment_records.id", ondelete="RESTRICT"), unique=True, index=True)


class DeploymentRecordTransfer(Base):
    __tablename__ = "deployment_record_transfers"
    transfer_id: Mapped[str] = mapped_column(ForeignKey("checkpoint_transfers.id", ondelete="RESTRICT"), primary_key=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("deployment_records.id", ondelete="RESTRICT"), index=True)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
