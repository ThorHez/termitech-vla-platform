from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ProfileInput(InputModel):
    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    code_dir: str = Field(min_length=1, max_length=4096)
    python: str = Field(min_length=1, max_length=4096)
    adapter: str = Field(min_length=1, max_length=100)
    recipe: str | None = Field(default=None, max_length=255)
    execution_mode: Literal["native", "docker"] = "native"
    docker_image: str | None = None
    metadata: dict = Field(default_factory=dict)


class RegisterInput(InputModel):
    name: str = Field(min_length=1, max_length=200)
    hostname: str = Field(min_length=1, max_length=255)
    roots: list[str] = Field(min_length=1, max_length=100)
    output_root: str
    profiles: list[ProfileInput] = Field(default_factory=list, max_length=100)


class GPUInput(InputModel):
    uuid: str = Field(min_length=1, max_length=200)
    name: str | None = None
    memory_total_mb: float | None = Field(default=None, ge=0)
    memory_used_mb: float | None = Field(default=None, ge=0)
    utilization_pct: float | None = Field(default=None, ge=0, le=100)
    processes: list[dict] | None = None


class ActiveAttemptInput(InputModel):
    attempt_id: str
    pid: int | None = None
    state: Literal["STARTING", "PREPARING", "RUNNING", "CANCELLING", "STOPPING", "UNKNOWN", "SUCCEEDED", "FAILED", "CANCELLED"]


class HeartbeatInput(InputModel):
    node_id: str
    gpus: list[GPUInput] = Field(default_factory=list, max_length=256)
    metrics: dict = Field(default_factory=dict)
    active_attempts: list[ActiveAttemptInput] = Field(default_factory=list, max_length=1000)
    profiles: list[ProfileInput] | None = Field(default=None, max_length=100)


class AssetInput(InputModel):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["DATASET", "MODEL", "NORM"]
    node_id: str
    path: str = Field(min_length=1, max_length=4096)
    version: str = Field(default="1", min_length=1, max_length=200)
    metadata: dict = Field(default_factory=dict)


class PreviewInput(InputModel):
    name: str = Field(min_length=1, max_length=200)
    node_id: str
    profile_id: str
    template_id: str
    inputs: dict[str, str] = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    gpu_count: int = Field(default=1, ge=0, le=256, strict=True)


class RunInput(InputModel):
    preflight_id: str


class AckInput(InputModel):
    node_id: str
    status: Literal["APPLIED", "REJECTED"]
    result: dict = Field(default_factory=dict)


class EventInput(InputModel):
    attempt_id: str
    seq: int = Field(ge=1, strict=True)
    type: Literal["STARTED", "METRIC", "LOG", "RESOURCE", "SUCCEEDED", "FAILED", "CANCELLED"]
    timestamp: str
    data: dict = Field(default_factory=dict)


class EventBatchInput(InputModel):
    node_id: str
    events: list[EventInput] = Field(max_length=1000)
