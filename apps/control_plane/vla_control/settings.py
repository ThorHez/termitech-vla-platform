"""Explicit configuration; development credentials are never built in."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./.state/control.db"
    admin_token: str = ""
    enrollment_token: str = ""
    legacy_admin_enabled: bool = False
    session_ttl_seconds: int = 28800
    session_idle_seconds: int = 3600
    session_reauth_seconds: int = 300
    session_cookie_secure: bool = True
    public_url: str = ""
    public_url_aliases: str = ""
    require_https: bool = False
    forwarded_allow_ips: str = "127.0.0.1,::1"
    tls_cert_file: str = ""
    tls_key_file: str = ""
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
            legacy_admin_enabled=os.getenv("VLA_LEGACY_ADMIN_ENABLED", "false").lower() in {"1", "true", "yes"},
            session_ttl_seconds=int(os.getenv("VLA_SESSION_TTL_SECONDS", "28800")),
            session_idle_seconds=int(os.getenv("VLA_SESSION_IDLE_SECONDS", "3600")),
            session_reauth_seconds=int(os.getenv("VLA_SESSION_REAUTH_SECONDS", "300")),
            session_cookie_secure=os.getenv("VLA_SESSION_COOKIE_SECURE", "true").lower() in {"1", "true", "yes"},
            public_url=os.getenv("VLA_PUBLIC_URL", ""),
            public_url_aliases=os.getenv("VLA_PUBLIC_URL_ALIASES", ""),
            require_https=os.getenv("VLA_REQUIRE_HTTPS", "false").lower() in {"1", "true", "yes"},
            forwarded_allow_ips=os.getenv("VLA_FORWARDED_ALLOW_IPS", cls.forwarded_allow_ips),
            tls_cert_file=os.getenv("VLA_TLS_CERT_FILE", ""),
            tls_key_file=os.getenv("VLA_TLS_KEY_FILE", ""),
            scheduler_enabled=os.getenv("VLA_SCHEDULER_ENABLED", "true").lower() in {"1", "true", "yes"},
            scheduler_interval_seconds=float(os.getenv("VLA_SCHEDULER_INTERVAL_SECONDS", "2")),
            heartbeat_timeout_seconds=int(os.getenv("VLA_HEARTBEAT_TIMEOUT_SECONDS", "90")),
        )

    def validate(self) -> None:
        if not self.enrollment_token:
            raise ValueError("VLA_ENROLLMENT_TOKEN must be configured")
        if self.legacy_admin_enabled and not self.admin_token:
            raise ValueError("VLA_ADMIN_TOKEN is required when VLA_LEGACY_ADMIN_ENABLED is enabled")
        if self.admin_token and self.admin_token == self.enrollment_token:
            raise ValueError("Administrator and enrollment credentials must be different")
        if not 300 <= self.session_ttl_seconds <= 86400:
            raise ValueError("Session lifetime must be between 300 and 86400 seconds")
        if not 60 <= self.session_idle_seconds <= self.session_ttl_seconds:
            raise ValueError("Session idle timeout must be between 60 seconds and the session lifetime")
        if not 30 <= self.session_reauth_seconds <= min(900, self.session_idle_seconds):
            raise ValueError("Sensitive-operation verification must be between 30 seconds and 900 seconds, within the idle timeout")
        if self.scheduler_interval_seconds <= 0 or self.heartbeat_timeout_seconds <= 0:
            raise ValueError("Scheduler and heartbeat intervals must be positive")
        from .transport_security import validate_transport_settings
        validate_transport_settings(self)
