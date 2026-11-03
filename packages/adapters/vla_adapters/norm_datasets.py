"""Independent normalization inputs; no training weights or old norm assets."""

if __package__:
    from .automatic_config import normalize_auto_config
    from .data_processors import normalize_processor
else:
    from automatic_config import normalize_auto_config
    from data_processors import normalize_processor


def normalize_norm_datasets(value):
    if value is None:
        return None
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("NORM_DATASETS_INVALID: 请配置 1 到 64 个独立统计的数据集")
    rows, seen = [], set()
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("NORM_DATASETS_INVALID: 每项需要数据集和数据处理方式")
        row = {key: item for key, item in row.items() if key not in {"recipe_index", "processor"} or item is not None}
        if set(row) not in ({"dataset_asset_id", "recipe_index"}, {"dataset_asset_id", "processor"}):
            raise ValueError("NORM_DATASETS_INVALID: 每项只能选择配方数据项或独立数据处理器；不能设置归一化资产或混采权重")
        asset_id = row["dataset_asset_id"]
        if not isinstance(asset_id, str) or not 1 <= len(asset_id) <= 200 or any(ord(c) < 32 for c in asset_id):
            raise ValueError("NORM_DATASETS_INVALID: 请选择有效的数据集资产")
        if asset_id in seen:
            raise ValueError("NORM_DATASETS_DUPLICATE: 同一数据集只能添加一次")
        seen.add(asset_id)
        if "recipe_index" in row:
            if type(row["recipe_index"]) is not int or not 0 <= row["recipe_index"] < 64:
                raise ValueError("NORM_DATASETS_INVALID: 配方数据项索引无效")
        else:
            row = {**row, "processor": normalize_processor(row["processor"], norm_stats=True)}
        rows.append(row)
    return rows


def norm_inputs(rows):
    normalized = normalize_norm_datasets(rows)
    return {} if normalized is None else {f"dataset_{index}": row["dataset_asset_id"]
                                          for index, row in enumerate(normalized)}


def validate_norm_configuration(auto_config, rows):
    """Automatic statistics need explicit processors, never training assets."""
    clean = normalize_auto_config(auto_config)
    if clean is None:
        return None
    if clean["finetune_mode"] != "full":
        raise ValueError("AUTO_CONFIG_UNSUPPORTED: 统计任务无需微调方式，请使用 full 模型结构配置")
    normalized = normalize_norm_datasets(rows)
    if not normalized or any("processor" not in row for row in normalized):
        raise ValueError("AUTO_CONFIG_PROCESSOR_REQUIRED: 自动统计配置要求每个数据集明确选择数据处理器或字段组合")
    return clean


def validate_norm_assets(rows, inputs):
    normalized = normalize_norm_datasets(rows)
    expected = {"dataset": None} if normalized is None else norm_inputs(normalized)
    if not isinstance(inputs, dict) or set(inputs) != set(expected) or any(
        not isinstance(inputs.get(role), dict) or inputs[role].get("kind") != "DATASET"
        or not inputs[role].get("path") or asset_id is not None and inputs[role].get("id") != asset_id
        for role, asset_id in expected.items()
    ):
        raise ValueError("NORM_DATASETS_INVALID: 统计数据集与本次输入资产不一致，不能包含模型、归一化资产或混采权重")
    return normalized
