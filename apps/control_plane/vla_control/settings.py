"""Explicit configuration; development credentials are never built in."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./.state/control.db"
    admin_token: str = ""
    enrollment_token: str = ""
    scheduler_enabled: bool = True
    scheduler_interval_seconds: float = 2.0
    heartbeat_timeout_seconds: int = 90
    preview_ttl_seconds: int = 900

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("VLA_DATABASE_URL", cls.database_url),
            admin_token=os.getenv("VLA_ADMIN_TOKEN", ""),
            enrollment_token=os.getenv("VLA_ENROLLMENT_TOKEN", ""),
            scheduler_enabled=os.getenv("VLA_SCHEDULER_ENABLED", "true").lower() in {"1", "true", "yes"},
            scheduler_interval_seconds=float(os.getenv("VLA_SCHEDULER_INTERVAL_SECONDS", "2")),
            heartbeat_timeout_seconds=int(os.getenv("VLA_HEARTBEAT_TIMEOUT_SECONDS", "90")),
        )

    def validate(self) -> None:
        if not self.admin_token or not self.enrollment_token:
            raise ValueError("VLA_ADMIN_TOKEN and VLA_ENROLLMENT_TOKEN must be configured")
        if self.admin_token == self.enrollment_token:
            raise ValueError("Administrator and enrollment credentials must be different")
        if self.scheduler_interval_seconds <= 0 or self.heartbeat_timeout_seconds <= 0:
            raise ValueError("Scheduler and heartbeat intervals must be positive")
