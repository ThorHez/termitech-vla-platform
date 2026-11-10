"""Reusable dataset registration formats, independent of assets."""

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, JSON, String, false
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class DatasetFormatTemplate(Base):
    __tablename__ = "dataset_format_templates"
    __table_args__ = (Index("uq_dataset_format_templates_user_name", "user_id", "name", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    profile: Mapped[dict] = mapped_column(JSON)
    shared: Mapped[bool] = mapped_column(Boolean, server_default=false())
    revision: Mapped[int] = mapped_column(BigInteger, default=1)
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40), index=True)
