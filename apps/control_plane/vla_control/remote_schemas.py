"""Administrator input for password-authenticated bare-metal installation."""

import ipaddress
import re

from pydantic import Field, SecretStr, field_validator

from .schemas import AgentConfigInput, InputModel


class SSHAuthenticationInput(InputModel):
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535, strict=True)
    username: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    password: SecretStr = Field(min_length=1, max_length=4096)

    @field_validator("host")
    @classmethod
    def plain_host(cls, value):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            if (not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value)
                    or ".." in value or any(not label or len(label) > 63 for label in value.split("."))):
                raise ValueError("Use a hostname or IP address without a scheme, path or credentials")
            return value.lower()


class SSHConnectionInput(SSHAuthenticationInput):
    install_dir: str = Field(min_length=2, max_length=4096)


class SSHPathSuggestInput(InputModel):
    prefix: str = Field(default="", max_length=4096)
    limit: int = Field(default=30, ge=1, le=50, strict=True)
    directory_only: bool = Field(default=True, strict=True)


class SSHInstallCheckInput(InputModel):
    install_dir: str = Field(min_length=2, max_length=4096)


class RemoteDeploymentInput(InputModel):
    connection_id: str = Field(min_length=1, max_length=36)
    install_dir: str = Field(min_length=2, max_length=4096)
    agent: AgentConfigInput
