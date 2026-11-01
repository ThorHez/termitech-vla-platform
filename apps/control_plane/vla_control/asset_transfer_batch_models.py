"""Account-owned receipts for atomic groups of existing asset transfers."""

from sqlalchemy import ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class AssetTransferBatch(Base):
    __tablename__ = "asset_transfer_batches"
    __table_args__ = (Index("uq_asset_transfer_batches_owner_key", "user_id", "request_key", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    request_key: Mapped[str] = mapped_column(String(200))
    request_digest: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))
