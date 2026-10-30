"""Validate the optional, node-bound GPU request without choosing replacements."""


def requested_gpu_uuids(value, gpu_count, available_uuids=None):
    if value is None:
        return None
    if (type(value) is not list or not 1 <= len(value) <= 256
            or any(type(uuid) is not str or not 1 <= len(uuid) <= 200
                   or any(character.isspace() or ord(character) < 32 or character == "," for character in uuid)
                   for uuid in value)
            or len(set(value)) != len(value)):
        raise ValueError("REQUESTED_GPU_INVALID: 指定 GPU 必须是非空、无重复的 UUID 列表")
    if len(value) != gpu_count:
        raise ValueError("REQUESTED_GPU_COUNT_MISMATCH: 指定 GPU 数量必须与申请 GPU 数量一致")
    if available_uuids is not None and not set(value) <= set(available_uuids):
        raise ValueError("REQUESTED_GPU_NOT_FOUND: 指定 GPU 不属于当前机器或已不可用，请重新选择并预检")
    return list(value)
