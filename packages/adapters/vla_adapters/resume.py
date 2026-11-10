"""Validate the controller-owned identity of one native SFT checkpoint."""

from pathlib import Path
import re


_FIELDS = {"asset_id", "run_id", "attempt_id", "path", "output_dir", "step"}


def normalize_resume_source(value, inputs, *, template_id, entrypoint=None):
    if value is None:
        return None
    if template_id != "openpi-sft" or entrypoint is not None:
        raise ValueError("RESUME_UNSUPPORTED: 仅原生单数据集 SFT 的标准训练入口支持断点续训")
    if (not isinstance(value, dict) or set(value) != _FIELDS
            or type(value["step"]) is not int or not 0 <= value["step"] < 2**53
            or any(not isinstance(value[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value[key])
                   for key in ("asset_id", "run_id", "attempt_id"))):
        raise ValueError("RESUME_SOURCE_INVALID: 续训来源记录不完整")
    for key in ("path", "output_dir"):
        raw = value[key]
        if (not isinstance(raw, str) or len(raw) > 4096 or not Path(raw).is_absolute()
                or str(Path(raw)) != raw or ".." in Path(raw).parts or raw.startswith("//")
                or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in raw)):
            raise ValueError("RESUME_SOURCE_INVALID: 续训来源必须使用规范的本机绝对路径")
    output, checkpoint = Path(value["output_dir"]), Path(value["path"])
    if (output.name != value["attempt_id"] or checkpoint.parent.parent.parent != output / "checkpoints"
            or checkpoint.parent.name != "platform" or checkpoint.name != str(value["step"])):
        raise ValueError("RESUME_SOURCE_INVALID: 续训检查点不属于所选来源任务和步数")
    model = inputs.get("model") if isinstance(inputs, dict) else None
    if (not isinstance(model, dict) or model.get("kind") != "MODEL"
            or model.get("id") != value["asset_id"] or model.get("path") != value["path"]):
        raise ValueError("RESUME_SOURCE_INVALID: 续训来源必须与所选模型检查点资产一致")
    return dict(value)
