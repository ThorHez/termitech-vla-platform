"""Versioned, declarative dataset semantics, independent of executable processors.

Profiles describe registered data. They are not a claim that the machine-local
files have been inspected, nor permission to execute a processing factory.
"""

from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from vla_adapters.composition_spec import canonical_text_trim, normalize_composition


class _ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True)


def _text(value):
    if isinstance(value, str):
        value = canonical_text_trim(value)
        if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
            raise ValueError("数据集描述不能包含控制字符")
    return value


class DatasetCamera(_ProfileModel):
    name: str = Field(min_length=1, max_length=100)
    key: str = Field(min_length=1, max_length=256)
    kind: Literal["rgb", "depth"]

    _normalize_text = field_validator("name", "key", mode="before")(_text)


class DatasetFieldAnnotation(_ProfileModel):
    """Labels describe a source field; they never prescribe a training layout."""
    key: str = Field(min_length=1, max_length=256)
    role: Literal["state", "action"]
    representation: Literal["eef", "joint", "gripper", "unknown"] = "unknown"
    value_mode: Literal["absolute", "delta", "unknown"] = "unknown"
    component: Literal["position", "rotation", "joint", "gripper", "other"] = "other"
    arm: Literal["left", "right", "shared", "unknown"] = "unknown"
    shape: list[Annotated[int, Field(ge=1, le=2**31 - 1)]] | None = Field(default=None, min_length=1, max_length=5)
    dtype: str | None = Field(default=None, min_length=1, max_length=64)
    names: list[Annotated[str, Field(min_length=1, max_length=256)]] | None = Field(default=None, max_length=512)
    horizon: int | None = Field(default=None, ge=1, le=2**31 - 1)

    _normalize_text = field_validator("key", "dtype", mode="before")(_text)

    @field_validator("names", mode="before")
    @classmethod
    def normalize_names(cls, value):
        return [_text(item) for item in value] if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_gripper_representation(self):
        if self.representation == "gripper" and self.component != "gripper":
            raise ValueError("夹爪开度表示只允许用于夹爪分量")
        return self

    @model_validator(mode="after")
    def validate_source_horizon(self):
        if "horizon" not in self.model_fields_set:
            return self
        if self.role != "action" or self.horizon is None:
            raise ValueError("horizon 只允许用于动作字段，并且必须是正整数")
        if self.shape is not None:
            expected = 1 if len(self.shape) == 1 else self.shape[0] if len(self.shape) == 2 else None
            if self.horizon != expected:
                raise ValueError("动作字段 horizon 必须与原始 [D] 或 [H,D] 形状一致")
        return self


class DatasetProfile(_ProfileModel):
    schema_version: Literal[1]
    format: Literal["lerobot", "rlds", "other"]
    robot_type: str = Field(min_length=1, max_length=100)
    arm_mode: Literal["single", "bimanual"]
    action_representation: Literal["eef", "joint", "unknown"]
    cameras: list[DatasetCamera] = Field(min_length=1, max_length=16)
    task_description: str = Field(min_length=1, max_length=2000)
    action_dim: int | None = Field(default=None, ge=1, le=512)
    state_dim: int | None = Field(default=None, ge=1, le=512)
    fps: float | None = Field(default=None, gt=0, le=1000)
    action_mode: Literal["absolute", "delta", "unknown"] = "unknown"
    rotation_representation: Literal["axis_angle", "quaternion", "rotation_6d", "none", "unknown"] = "unknown"
    feature_keys: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(default_factory=list, max_length=256)
    field_annotations: list[DatasetFieldAnnotation] | None = Field(default=None, max_length=256)
    processing: dict | None = None

    @field_validator("processing", mode="before")
    @classmethod
    def composed_processing(cls, value):
        return normalize_composition(value) if value is not None else None

    @field_validator("schema_version", mode="before")
    @classmethod
    def version_is_an_integer(cls, value):
        if type(value) is not int:
            raise ValueError("数据集描述版本必须是整数 1")
        return value

    _normalize_text = field_validator("robot_type", mode="before")(_text)

    @field_validator("task_description", mode="before")
    @classmethod
    def description_text(cls, value):
        if isinstance(value, str):
            value = canonical_text_trim(value)
            if any((ord(char) < 32 and char not in "\n\t") or 127 <= ord(char) <= 159 for char in value):
                raise ValueError("任务描述不能包含控制字符")
        return value

    @field_validator("feature_keys", mode="before")
    @classmethod
    def feature_text(cls, value):
        return [_text(key) for key in value] if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_fields(self):
        if len({camera.name for camera in self.cameras}) != len(self.cameras):
            raise ValueError("相机名称不能重复")
        if len({camera.key for camera in self.cameras}) != len(self.cameras):
            raise ValueError("相机字段不能重复")
        if len(set(self.feature_keys)) != len(self.feature_keys):
            raise ValueError("数据字段不能重复")
        if self.field_annotations is not None:
            keys = [row.key for row in self.field_annotations]
            if len(set(keys)) != len(keys):
                raise ValueError("源字段标签不能重复")
            if self.feature_keys and not set(keys) <= set(self.feature_keys):
                raise ValueError("源字段标签必须对应已登记的真实字段")
        # Legacy processing is a historical artifact, not the authority for
        # source labels or the next training's independently selected layout.
        return self


def normalize_dataset_profile(value) -> dict:
    """Validate a complete v1 description without importing training code."""
    return DatasetProfile.model_validate(value).model_dump(mode="json", exclude_none=True)


def validate_asset_metadata(kind, metadata) -> dict:
    """Keep existing metadata intact; reserve one validated dataset namespace."""
    clean = deepcopy(metadata)
    if "dataset_profile" in clean:
        if kind != "DATASET":
            raise ValueError("只有数据集资产可以登记数据集描述")
        clean["dataset_profile"] = normalize_dataset_profile(clean["dataset_profile"])
    return clean
