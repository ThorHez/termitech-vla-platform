"""Account-owned metadata discovery and durable batch registration receipts."""

from sqlalchemy import Boolean, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class DatasetScan(Base):
    __tablename__ = "dataset_scans"
    __table_args__ = (Index("ix_dataset_scans_owner_created", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="RESTRICT"), index=True)
    path: Mapped[str] = mapped_column(Text)
    recursive: Mapped[bool] = mapped_column(Boolean)
    max_depth: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(24), index=True)
    binding_snapshot: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    expires_at: Mapped[str] = mapped_column(String(40), index=True)


class DatasetBatchRegistration(Base):
    __tablename__ = "dataset_batch_registrations"
    __table_args__ = (Index("uq_dataset_batches_owner_key", "user_id", "request_key", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    scan_id: Mapped[str] = mapped_column(ForeignKey("dataset_scans.id", ondelete="RESTRICT"), index=True)
    request_key: Mapped[str] = mapped_column(String(200))
    request_digest: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))
