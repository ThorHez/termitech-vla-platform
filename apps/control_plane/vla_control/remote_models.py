"""Remote installation metadata. SSH passwords are encrypted outside API views."""

from sqlalchemy import ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class RemoteConnection(Base):
    __tablename__ = "remote_connections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    host: Mapped[str] = mapped_column(String(255))
    port: Mapped[int]
    username: Mapped[str] = mapped_column(String(64))
    encrypted_password: Mapped[str] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(255))
    probe_result: Mapped[dict] = mapped_column(JSON)
    install_dir: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40))
    expires_at: Mapped[str] = mapped_column(String(40))


class RemoteDeployment(Base):
    __tablename__ = "remote_deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    connection_id: Mapped[str] = mapped_column(ForeignKey("remote_connections.id", ondelete="RESTRICT"), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    request_digest: Mapped[str] = mapped_column(String(64))
    target_key: Mapped[str] = mapped_column(String(64), index=True)
    onboarding_id: Mapped[str] = mapped_column(String(36), index=True)
    node_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    stage: Mapped[str] = mapped_column(String(64))
    logs: Mapped[list] = mapped_column(JSON, default=list)
    history: Mapped[list] = mapped_column(JSON, default=list)
    agent_config: Mapped[dict] = mapped_column(JSON)
    install_dir: Mapped[str] = mapped_column(Text)
    control_plane_url: Mapped[str] = mapped_column(Text)
    connection_mode: Mapped[str] = mapped_column(String(32))
    tunnel_state: Mapped[str] = mapped_column(String(32))
    tunnel_remote_port: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    started_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
