"""Explicit ordered data/norm pairs shared by the API, Agent and training Python."""

import math

if __package__:
    from .data_processors import normalize_processor
else:
    from data_processors import normalize_processor


def normalize_dataset_bindings(value):
    if value is None:
        return None
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("DATASET_BINDINGS_INVALID: 请配置 1 到 64 组数据集与归一化参数")
    result = []
    common = {"dataset_asset_id", "norm_asset_id", "weight"}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("DATASET_BINDINGS_INVALID: 每组需要数据集、归一化参数、数据处理配置与混采权重")
        # Optional Pydantic fields serialize as null unless explicitly excluded.
        # Keep the historical recipe-index row shape and digest unchanged.
        row = {key: item for key, item in row.items() if key not in {"recipe_index", "processor"} or item is not None}
        if set(row) not in (common | {"recipe_index"}, common | {"processor"}):
            raise ValueError("DATASET_BINDINGS_INVALID: 每组必须选择独立数据处理器或配方数据项，两者不能同时设置")
        if any(not isinstance(row[key], str) or not 1 <= len(row[key]) <= 200
               for key in ("dataset_asset_id", "norm_asset_id")):
            raise ValueError("DATASET_BINDINGS_INVALID: 请选择每组的数据集与归一化参数资产")
        if "recipe_index" in row:
            if type(row["recipe_index"]) is not int or not 0 <= row["recipe_index"] < 64:
                raise ValueError("DATASET_BINDINGS_INVALID: 数据处理配置索引无效")
        else:
            row = {**row, "processor": normalize_processor(row["processor"])}
        weight = row["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("DATASET_BINDINGS_INVALID: 混采权重必须是大于零的有限数值")
        result.append({**row, "weight": float(weight)})
    if not math.isfinite(sum(row["weight"] for row in result)):
        raise ValueError("DATASET_BINDINGS_INVALID: 混采权重总和超出数值范围")
    return result


def binding_inputs(bindings):
    """Asset IDs for all ordered roles; include every pair in run provenance."""
    rows = normalize_dataset_bindings(bindings)
    return {} if rows is None else {role: row[key] for i, row in enumerate(rows)
        for role, key in ((f"dataset_{i}", "dataset_asset_id"), (f"norm_{i}", "norm_asset_id"))}


def validate_binding_assets(bindings, inputs):
    rows = normalize_dataset_bindings(bindings)
    if rows is not None:
        expected = binding_inputs(rows)
        if set(inputs) != {"model", *expected} or any(
                not isinstance(inputs.get(role), dict) or inputs[role].get("id") != asset_id
                for role, asset_id in expected.items()):
            raise ValueError("DATASET_BINDINGS_INVALID: 数据集绑定与本次输入资产不一致，请重新预检")
    return rows
