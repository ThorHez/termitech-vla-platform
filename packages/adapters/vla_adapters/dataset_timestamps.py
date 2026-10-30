"""Read-only, bounded proof of the timestamp invariant used by LeRobot.

Only the two scalar parquet columns are streamed. No dataset builder, model,
video decoder, cache writer or GPU is constructed by this module.
"""

import ast
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import textwrap
import time

MODULE = "lerobot.common.datasets.lerobot_dataset"
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_FOOTER_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 20000
MAX_ROWS = 25_000_000
SCAN_SECONDS = 30
BATCH_ROWS = 8192
MAX_ERRORS = 5


def _fail(message, code="DATASET_TIMESTAMPS_INVALID"):
    raise ValueError(code + ": " + message)


def _limit(deadline):
    if time.monotonic() > deadline:
        _fail("真实时间戳检查超时，未完成校验，不能开始训练", "DATASET_TIMESTAMP_CHECK_LIMIT")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _managed(value, spec, *, inside=None):
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute() or ".." in Path(value).parts:
        _fail("时间戳数据源需要受管目录内的明确绝对路径")
    source = Path(value)
    if source.is_symlink():
        _fail("时间戳数据源不能使用符号链接：" + str(source))
    path = source.resolve(strict=True)
    roots = spec.get("allowed_roots")
    if (not isinstance(roots, list) or not roots
            or any(not isinstance(root, (str, Path)) or not Path(root).is_absolute() or ".." in Path(root).parts for root in roots)
            or not any(path.is_relative_to(Path(root).resolve(strict=True)) for root in roots)):
        _fail("时间戳数据源超出 Agent 受管目录：" + str(path))
    if inside is not None:
        if not source.is_relative_to(inside) or not path.is_relative_to(inside):
            _fail("时间戳文件超出本次数据集目录：" + str(source))
        current = source
        while current != inside:
            if current.is_symlink():
                _fail("时间戳数据源不能使用符号链接：" + str(current))
            current = current.parent
    return path


def _open_regular(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        _fail("时间戳数据源必须为普通文件：" + str(path))
    return os.fdopen(descriptor, "rb")


def _read(path, spec, root):
    source = _managed(path, spec, inside=root)
    with _open_regular(source) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > MAX_METADATA_BYTES:
            _fail("时间戳元信息超过读取上限", "DATASET_TIMESTAMP_CHECK_LIMIT")
        content = stream.read(MAX_METADATA_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(content) > MAX_METADATA_BYTES:
        _fail("时间戳元信息超过读取上限", "DATASET_TIMESTAMP_CHECK_LIMIT")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        _fail("时间戳元信息在检查时发生变化，请重新预检", "DATASET_TIMESTAMP_SOURCE_CHANGED")
    return content, hashlib.sha256(content).hexdigest()


def _json(content):
    return json.loads(content, parse_constant=lambda _: _fail("元信息包含非有限数值"))


def _source(function):
    path = Path(inspect.getsourcefile(function) or "").resolve(strict=True)
    with _open_regular(path) as stream:
        if os.fstat(stream.fileno()).st_size > MAX_SOURCE_BYTES:
            _fail("LeRobot 时间戳校验来源过大", "DATASET_TIMESTAMP_CHECK_UNAVAILABLE")
        content = stream.read(MAX_SOURCE_BYTES + 1)
    if len(content) > MAX_SOURCE_BYTES:
        _fail("LeRobot 时间戳校验来源超过读取上限", "DATASET_TIMESTAMP_CHECK_UNAVAILABLE")
    return str(path), hashlib.sha256(content).hexdigest()


def _runtime():
    try:
        module = importlib.import_module(MODULE)
        constructor = module.LeRobotDataset.__init__
        validator = module.check_timestamps_sync
        if not inspect.isfunction(constructor) or not inspect.isfunction(validator):
            raise ValueError("dynamic runtime")
        signature = inspect.signature(constructor)
        tolerance = signature.parameters["tolerance_s"].default
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("invalid tolerance default")
        tree = ast.parse(textwrap.dedent(inspect.getsource(validator)))
        for name, expression in (("diffs", "np.diff(timestamps)"),
                                 ("within_tolerance", "np.abs(diffs - (1.0 / fps)) <= tolerance_s")):
            expected = ast.parse(expression, mode="eval").body
            if not any(isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
                       and ast.dump(node.value, include_attributes=False) == ast.dump(expected, include_attributes=False)
                       for node in ast.walk(tree)):
                raise ValueError("unsupported timestamp formula")
        module_path, module_sha = _source(constructor)
        validator_path, validator_sha = _source(validator)
        return {"module": MODULE, "module_path": module_path, "module_sha256": module_sha,
                "constructor_signature": str(signature), "validator": "check_timestamps_sync",
                "validator_path": validator_path, "validator_sha256": validator_sha}, float(tolerance)
    except (ImportError, AttributeError, KeyError, OSError, TypeError, ValueError, SyntaxError) as exc:
        if str(exc).startswith("DATASET_"):
            raise
        raise ValueError("DATASET_TIMESTAMP_CHECK_UNAVAILABLE: 无法核验训练环境 LeRobot 的时间戳公式和默认容差，请检查训练环境") from exc


def _manifest(directory, spec, deadline):
    root = _managed(directory, spec)
    rows, pending, count = [], [(root, 0)], 0
    while pending:
        directory, depth = pending.pop()
        if depth > 16:
            _fail("数据目录层级超过检查上限", "DATASET_TIMESTAMP_CHECK_LIMIT")
        with os.scandir(directory) as entries:
            for entry in entries:
                _limit(deadline)
                count += 1
                if count > MAX_ENTRIES:
                    _fail("数据文件数量超过检查上限", "DATASET_TIMESTAMP_CHECK_LIMIT")
                if entry.is_symlink():
                    _fail("数据目录包含符号链接：" + entry.path)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append((Path(entry.path), depth + 1))
                elif stat.S_ISREG(info.st_mode):
                    rows.append([str(Path(entry.path).relative_to(root)), info.st_size, info.st_mtime_ns])
                else:
                    _fail("数据目录包含特殊文件：" + entry.path)
    rows.sort()
    return rows, _fingerprint(rows)


def _episodes(content, expected_count, expected_frames, deadline):
    episodes = {}
    for line in content.splitlines():
        _limit(deadline)
        if len(line) > 65536:
            _fail("episode 元信息单行过大", "DATASET_TIMESTAMP_CHECK_LIMIT")
        item = _json(line)
        if not isinstance(item, dict):
            _fail("episode 元信息需要 JSON 对象")
        index, length = item.get("episode_index"), item.get("length")
        if type(index) is not int or index < 0 or index in episodes or type(length) is not int or length <= 0:
            _fail("episode 元信息的编号、长度或唯一性无效")
        episodes[index] = length
        if len(episodes) > MAX_ENTRIES:
            _fail("episode 数量超过检查上限", "DATASET_TIMESTAMP_CHECK_LIMIT")
    if len(episodes) != expected_count or sum(episodes.values()) != expected_frames:
        _fail("meta/episodes.jsonl 的 episode 数量或总帧数与本次元信息不一致")
    return sorted(episodes.items())


def _parquet_file(path, parquet):
    stream = _open_regular(path)
    try:
        info = os.fstat(stream.fileno())
        if info.st_size < 12 or stream.read(4) != b"PAR1":
            _fail("无法读取真实 parquet 文件：" + str(path))
        stream.seek(-8, os.SEEK_END)
        tail = stream.read(8)
        footer_size = int.from_bytes(tail[:4], "little")
        if tail[4:] != b"PAR1" or not 0 < footer_size <= min(MAX_FOOTER_BYTES, info.st_size - 12):
            _fail("parquet footer 无效或超过有界读取上限：" + str(path))
        stream.seek(0)
        source = parquet.ParquetFile(stream)
        return stream, source, (info.st_size, info.st_mtime_ns)
    except BaseException:
        stream.close()
        raise


def inspect_dataset_timestamps(data, spec):
    """Verify every timestamp pair under the exact effective FPS declaration."""
    deadline = time.monotonic() + SCAN_SECONDS
    try:
        root = _managed(data["path"], spec)
        fps, frames, episode_count = data.get("effective_fps"), data.get("frames"), data.get("episodes")
        if type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= 1000:
            _fail("真实时间戳检查需要有效的本次生效 FPS")
        if type(frames) is not int or frames <= 0 or type(episode_count) is not int or episode_count <= 0:
            _fail("真实时间戳检查需要有效帧数和 episode 数量")
        if frames > MAX_ROWS:
            _fail("数据行数超过全量时间戳检查上限，不能视为校验通过", "DATASET_TIMESTAMP_CHECK_LIMIT")
        raw_info, info_sha = _read(root / "meta/info.json", spec, root)
        if data.get("metadata_sha256") != info_sha:
            _fail("数据集元信息在检查前发生变化，请重新预检", "DATASET_TIMESTAMP_SOURCE_CHANGED")
        info = _json(raw_info)
        if not isinstance(info, dict):
            _fail("info.json 需要 JSON 对象")
        if info.get("total_frames") != frames or info.get("total_episodes") != episode_count:
            _fail("本次帧数或 episode 数量与真实 info.json 不一致")
        raw_episodes, episodes_sha = _read(root / "meta/episodes.jsonl", spec, root)
        episodes = _episodes(raw_episodes, episode_count, frames, deadline)
        files, manifest_sha = _manifest(root / "data", spec, deadline)
        if data.get("data_manifest_digest") != manifest_sha:
            _fail("parquet 文件清单已变化，请重新预检", "DATASET_TIMESTAMP_SOURCE_CHANGED")
        pattern, chunks = info.get("data_path"), info.get("chunks_size")
        if not isinstance(pattern, str) or type(chunks) is not int or chunks <= 0:
            _fail("LeRobot 数据路径模板或 chunk 大小无效")
        selected, seen = [], set()
        for index, _ in episodes:
            _limit(deadline)
            relative = Path(pattern.format(episode_index=index, episode_chunk=index // chunks))
            if relative.is_absolute() or ".." in relative.parts:
                _fail("LeRobot 数据路径模板超出本机数据集")
            target = _managed(root / relative, spec, inside=root / "data")
            if target not in seen:
                seen.add(target)
                selected.append(target)
        actual = {root / "data" / row[0] for row in files if row[0].endswith(".parquet")}
        if not selected or set(selected) != actual:
            _fail("实际 parquet 文件与 episode 路径映射不一致，不能证明加载顺序")
        runtime, tolerance = _runtime()
        try:
            np = importlib.import_module("numpy")
            parquet = importlib.import_module("pyarrow.parquet")
        except ImportError as exc:
            raise ValueError("DATASET_TIMESTAMP_CHECK_UNAVAILABLE: 训练环境需要 NumPy 与 PyArrow 才能检查真实时间戳") from exc
        checked_rows = checked_pairs = episode_cursor = episode_rows = 0
        previous = None
        for file in selected:
            _limit(deadline)
            stream, source, before = _parquet_file(file, parquet)
            try:
                names = source.schema_arrow.names
                if names.count("timestamp") != 1 or names.count("episode_index") != 1:
                    _fail("parquet 缺少 timestamp 或 episode_index 实际列：" + str(file))
                integers = {kind + str(width) for kind in ("int", "uint") for width in (8, 16, 32, 64)}
                if (str(source.schema_arrow.field("timestamp").type) not in integers | {"halffloat", "float", "double"}
                        or str(source.schema_arrow.field("episode_index").type) not in integers):
                    _fail("timestamp/episode_index 必须是数值标量列，不能读取嵌套、字符串或时间对象：" + str(file))
                if source.metadata.num_rows <= 0 or source.metadata.num_rows > MAX_ROWS:
                    _fail("parquet 行数无效或超过检查上限")
                for batch in source.iter_batches(batch_size=BATCH_ROWS, columns=["timestamp", "episode_index"], use_threads=False):
                    _limit(deadline)
                    values = {name: batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
                              for name in ("timestamp", "episode_index")}
                    timestamps, indices = values["timestamp"], values["episode_index"]
                    if (timestamps.ndim != 1 or indices.ndim != 1 or timestamps.shape != indices.shape
                            or timestamps.dtype.kind not in "iuf" or indices.dtype.kind not in "iu"
                            or not np.all(np.isfinite(timestamps))):
                        _fail("timestamp 必须是有限数值标量，episode_index 必须是整数标量：" + str(file))
                    offset = 0
                    while offset < len(timestamps):
                        if episode_cursor >= len(episodes):
                            _fail("实际 parquet 行数超过 episode 元信息")
                        index, length = episodes[episode_cursor]
                        take = min(length - episode_rows, len(timestamps) - offset)
                        segment = timestamps[offset:offset + take]
                        if not np.all(indices[offset:offset + take] == index):
                            _fail("parquet 的 episode_index、顺序或长度与 meta/episodes.jsonl 不一致：" + str(file))
                        diffs = np.diff(segment)
                        if previous is not None:
                            diffs = np.concatenate((np.asarray([segment[0] - previous]), diffs))
                        within = np.abs(diffs - (1.0 / fps)) <= tolerance
                        bad = np.flatnonzero(~within)
                        if len(bad):
                            samples = [{"episode_index": index, "interval_seconds": float(diffs[i])}
                                       for i in bad[:MAX_ERRORS]]
                            _fail(f"数据集 {str(root)[:512]} 的真实时间戳与生效 {fps:g} Hz 不一致："
                                  f"期望帧间隔 {1.0 / fps:.9g} 秒，容差 {tolerance:.9g} 秒；"
                                  f"实测样例（最多 {MAX_ERRORS} 条）{json.dumps(samples, ensure_ascii=False)}。"
                                  "仅修改 FPS 声明不会重采样；请按真实采样率登记，或先正确重采样数据")
                        checked_pairs += len(diffs)
                        checked_rows += take
                        if checked_rows > MAX_ROWS:
                            _fail("实际数据行数超过检查上限", "DATASET_TIMESTAMP_CHECK_LIMIT")
                        previous = segment[-1].copy()
                        offset += take
                        episode_rows += take
                        if episode_rows == length:
                            episode_cursor += 1
                            episode_rows = 0
                            previous = None
                after = os.fstat(stream.fileno())
                if before != (after.st_size, after.st_mtime_ns):
                    _fail("parquet 文件在读取期间发生变化，请重新预检", "DATASET_TIMESTAMP_SOURCE_CHANGED")
            finally:
                stream.close()
        if checked_rows != frames or episode_cursor != len(episodes) or episode_rows:
            _fail("真实 parquet 帧数或 episode 长度与元信息不一致")
        _, final_info = _read(root / "meta/info.json", spec, root)
        _, final_episodes = _read(root / "meta/episodes.jsonl", spec, root)
        _, final_manifest = _manifest(root / "data", spec, deadline)
        if (final_info, final_episodes, final_manifest) != (info_sha, episodes_sha, manifest_sha):
            _fail("数据集在时间戳检查期间发生变化，请重新预检", "DATASET_TIMESTAMP_SOURCE_CHANGED")
        return {"schema_version": 1, "status": "verified", "scope": "all parquet timestamp/episode_index rows",
                "effective_fps": fps, "expected_interval_seconds": 1.0 / fps, "tolerance_s": tolerance,
                "checked_rows": checked_rows, "checked_pairs": checked_pairs, "checked_episodes": len(episodes),
                "parquet_files": len(selected), "metadata_sha256": info_sha, "episodes_sha256": episodes_sha,
                "data_manifest_digest": manifest_sha, "runtime": runtime}
    except (OSError, KeyError, IndexError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        if str(exc).startswith("DATASET_"):
            raise
        raise ValueError("DATASET_TIMESTAMPS_INVALID: 无法完成真实时间戳校验：" + str(exc)[:1000]) from exc
