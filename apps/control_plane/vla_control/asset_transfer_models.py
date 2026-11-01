"""Persistent receipts for byte-exact copies of registered machine assets."""

from sqlalchemy import JSON, BigInteger, Boolean, Float, ForeignKey, Integer, String, Text, false
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


ASSET_TRANSFER_WORKING = frozenset({'QUEUED', 'PREPARING', 'TRANSFERRING', 'VERIFYING'})
ASSET_TRANSFER_ACTIVE = ASSET_TRANSFER_WORKING | {'CANCELLING'}
ASSET_TRANSFER_RETRYABLE = frozenset({'FAILED', 'CANCELLED', 'INTERRUPTED'})
ASSET_TRANSFER_TERMINAL = frozenset({'SUCCEEDED', *ASSET_TRANSFER_RETRYABLE})


class AssetTransfer(Base):
    __tablename__ = 'asset_transfers'

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_key: Mapped[str] = mapped_column(String(200), unique=True)
    request_digest: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    source_asset_id: Mapped[str] = mapped_column(ForeignKey('assets.id', ondelete='RESTRICT'), index=True)
    source_node_id: Mapped[str] = mapped_column(ForeignKey('nodes.id', ondelete='RESTRICT'), index=True)
    source_path: Mapped[str] = mapped_column(Text)
    source_kind: Mapped[str] = mapped_column(String(20))
    target_node_id: Mapped[str] = mapped_column(ForeignKey('nodes.id', ondelete='RESTRICT'), index=True)
    target_directory: Mapped[str] = mapped_column(Text)
    target_name: Mapped[str] = mapped_column(String(160))
    target_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    asset_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_asset_id: Mapped[str | None] = mapped_column(ForeignKey('assets.id', ondelete='RESTRICT'), nullable=True)
    include_norm_stats: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    norm_sync: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    bytes_total: Mapped[int] = mapped_column(BigInteger, default=0)
    bytes_transferred: Mapped[int] = mapped_column(BigInteger, default=0)
    files_total: Mapped[int] = mapped_column(Integer, default=0)
    files_transferred: Mapped[int] = mapped_column(Integer, default=0)
    speed_bytes_per_second: Mapped[float] = mapped_column(Float, default=0)
    progress_pct: Mapped[float] = mapped_column(Float, default=0)
    logs: Mapped[str] = mapped_column(Text, default='')
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manifest: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    binding_snapshot: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    finished_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    deleted_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
