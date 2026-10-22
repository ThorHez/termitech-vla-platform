"""Platform-owned recipe identity and immutable executable revisions."""

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class SavedRecipe(Base):
    __tablename__ = "saved_recipes"
    __table_args__ = (
        ForeignKeyConstraint(["node_id", "profile_id"], ["profiles.node_id", "profiles.id"], ondelete="RESTRICT"),
        UniqueConstraint("node_id", "profile_id", "name"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(36), index=True)
    profile_id: Mapped[str] = mapped_column(String(200), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    latest_revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str] = mapped_column(String(40))


class RecipeRevision(Base):
    __tablename__ = "recipe_revisions"
    __table_args__ = (UniqueConstraint("recipe_id", "revision"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    recipe_id: Mapped[str] = mapped_column(ForeignKey("saved_recipes.id", ondelete="RESTRICT"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    definition: Mapped[dict] = mapped_column(JSON)
    source_detail: Mapped[dict] = mapped_column(JSON)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40))
