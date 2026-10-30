"""The preflight timestamp contract reads source columns without constructing a loader."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters import dataset_timestamps as check


RUNTIME = '''
class LeRobotDataset:
    def __init__(self, repo_id, tolerance_s=1e-4):
        raise AssertionError("A preflight must not construct the dataset")
def check_timestamps_sync(timestamps, episode_indices, episode_data_index, fps, tolerance_s, raise_value_error=True):
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s
    return within_tolerance.all()
'''


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    path = tmp_path / "verified_lerobot.py"
    path.write_text(RUNTIME)
    spec = importlib.util.spec_from_file_location(check.MODULE, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, check.MODULE, module)
    spec.loader.exec_module(module)
    return module


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fake_write(path, timestamps, indices, **extra):
    payload = json.dumps({"timestamp": timestamps, "episode_index": indices, **extra}).encode()
    path.write_bytes(b"PAR1" + payload + b"x" + (1).to_bytes(4, "little") + b"PAR1")


def dataset(tmp_path, episodes=None, *, fps=20, writer=None, pattern="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"):
    episodes = episodes if episodes is not None else [(0, [0, .05, .1]), (1, [0, .05, .1])]
    root = tmp_path / "dataset"
    info = {"fps": 60, "data_path": pattern, "chunks_size": 1000,
            "total_frames": sum(len(times) for _, times in episodes), "total_episodes": len(episodes)}
    _json_write(root / "meta/info.json", info)
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": index, "length": len(times)}) for index, times in episodes) + "\n")
    for index, times in episodes:
        path = root / pattern.format(episode_index=index, episode_chunk=index // 1000)
        path.parent.mkdir(parents=True, exist_ok=True)
        (writer or _fake_write)(path, times, [index] * len(times))
    spec = {"allowed_roots": [str(tmp_path)]}
    _, manifest = check._manifest(root / "data", spec, float("inf"))
    data = {"path": str(root), "frames": info["total_frames"], "episodes": len(episodes),
            "effective_fps": fps, "metadata_sha256": _hash(root / "meta/info.json"),
            "data_manifest_digest": manifest}
    return data, spec


def _update_manifest(data, spec):
    data["data_manifest_digest"] = check._manifest(Path(data["path"]) / "data", spec, float("inf"))[1]


@pytest.fixture
def parquet(runtime, monkeypatch):
    np = pytest.importorskip("numpy")

    class Column:
        def __init__(self, values, dtype):
            self.values, self.dtype = values, dtype

        def to_numpy(self, zero_copy_only=False):
            assert zero_copy_only is False
            return np.asarray(self.values, dtype=self.dtype)

    class Batch:
        def __init__(self, values, names):
            self.values, self.names = values, names
            self.schema = SimpleNamespace(get_field_index=names.index)

        def column(self, index):
            name = self.names[index]
            return Column(self.values[name], "float32" if name == "timestamp" else self.values.get("episode_dtype", "int64"))

    class Source:
        def __init__(self, stream):
            stream.seek(0)
            self.values = json.loads(stream.read()[4:-9])
            self.names = self.values.get("columns", ["timestamp", "episode_index"])
            types = {"timestamp": self.values.get("timestamp_type", "float"), "episode_index": "int64"}
            self.schema_arrow = SimpleNamespace(names=self.names, field=lambda name: SimpleNamespace(type=types[name]))
            self.metadata = SimpleNamespace(num_rows=len(self.values["timestamp"]))

        def iter_batches(self, batch_size, columns, use_threads):
            assert columns == ["timestamp", "episode_index"] and use_threads is False
            for offset in range(0, len(self.values["timestamp"]), batch_size):
                values = {name: self.values[name][offset:offset + batch_size] for name in columns}
                values["episode_dtype"] = self.values.get("episode_dtype", "int64")
                yield Batch(values, columns)

    module = ModuleType("pyarrow.parquet")
    module.ParquetFile = Source
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", module)
    return module


def test_full_read_matches_registered_rate_without_changing_metadata_or_sources(tmp_path, parquet):
    data, spec = dataset(tmp_path)
    before = {str(path): _hash(path) for path in Path(data["path"]).rglob("*") if path.is_file()}
    original = deepcopy(data)
    result = check.inspect_dataset_timestamps(data, spec)
    assert result == check.inspect_dataset_timestamps(data, spec)
    assert result["effective_fps"] == 20 and result["tolerance_s"] == 1e-4
    assert result["checked_rows"] == 6 and result["checked_pairs"] == 4
    assert result["checked_episodes"] == result["parquet_files"] == 2
    assert result["runtime"]["validator"] == "check_timestamps_sync"
    assert result["status"] == "verified" and data == original
    assert before == {str(path): _hash(path) for path in Path(data["path"]).rglob("*") if path.is_file()}
    assert json.loads((Path(data["path"]) / "meta/info.json").read_text())["fps"] == 60


def test_actual_60hz_rows_reject_a_20hz_declaration_with_at_most_five_examples(tmp_path, parquet):
    data, spec = dataset(tmp_path, [(0, [index / 60 for index in range(30)])])
    with pytest.raises(ValueError, match="DATASET_TIMESTAMPS_INVALID") as failure:
        check.inspect_dataset_timestamps(data, spec)
    message = str(failure.value)
    assert "20 Hz" in message and "0.05" in message and "0.01666" in message
    assert "重采样" in message and message.count('"episode_index"') == 5
    assert len(message) < 2000


def test_boundaries_skip_episode_resets_but_check_pairs_across_record_batches(tmp_path, parquet, monkeypatch):
    monkeypatch.setattr(check, "BATCH_ROWS", 2)
    data, spec = dataset(tmp_path, [(0, [0, .05, .1]), (1, [0, .05, .1])])
    assert check.inspect_dataset_timestamps(data, spec)["checked_pairs"] == 4
    file = Path(data["path"]) / "data/chunk-000/episode_000000.parquet"
    _fake_write(file, [0, .05, .3], [0, 0, 0])
    _update_manifest(data, spec)
    with pytest.raises(ValueError, match="真实时间戳"):
        check.inspect_dataset_timestamps(data, spec)


def test_episode_order_is_numeric_even_for_unpadded_filenames_and_crosses_chunks(tmp_path, parquet):
    pattern = "data/chunk-{episode_chunk}/episode_{episode_index}.parquet"
    data, spec = dataset(tmp_path, [(9, [0, .05]), (10, [0, .05]), (1000, [0, .05])], pattern=pattern)
    result = check.inspect_dataset_timestamps(data, spec)
    assert result["checked_rows"] == 6 and result["checked_pairs"] == 3


@pytest.mark.parametrize("fps", [None, 0, -1, True, "20", float("nan"), float("inf"), 1001])
def test_invalid_effective_fps_never_silently_skips_validation(tmp_path, fps):
    data, spec = dataset(tmp_path, fps=fps)
    with pytest.raises(ValueError, match="生效 FPS"):
        check.inspect_dataset_timestamps(data, spec)


def test_resource_budget_exhaustion_is_not_reported_as_success(tmp_path, monkeypatch):
    data, spec = dataset(tmp_path)
    monkeypatch.setattr(check, "MAX_ROWS", 5)
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_CHECK_LIMIT"):
        check.inspect_dataset_timestamps(data, spec)


def test_scan_deadline_is_enforced_before_loading_optional_dependencies(tmp_path, monkeypatch):
    data, spec = dataset(tmp_path)
    readings = iter([0, check.SCAN_SECONDS + 1])
    monkeypatch.setattr(check.time, "monotonic", lambda: next(readings, check.SCAN_SECONDS + 1))
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_CHECK_LIMIT"):
        check.inspect_dataset_timestamps(data, spec)


def test_metadata_and_manifest_mutations_require_a_new_preflight(tmp_path):
    data, spec = dataset(tmp_path)
    path = Path(data["path"]) / "meta/info.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_SOURCE_CHANGED"):
        check.inspect_dataset_timestamps(data, spec)
    data["metadata_sha256"] = _hash(path)
    file = next((Path(data["path"]) / "data").rglob("*.parquet"))
    file.write_bytes(file.read_bytes() + b"x")
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_SOURCE_CHANGED"):
        check.inspect_dataset_timestamps(data, spec)


@pytest.mark.parametrize("mode", ["root", "file", "directory", "outside"])
def test_symlinks_and_out_of_scope_sources_fail_closed(tmp_path, mode):
    data, spec = dataset(tmp_path)
    root = Path(data["path"])
    if mode == "root":
        link = tmp_path / "linked";link.symlink_to(root, target_is_directory=True);data["path"] = str(link)
    elif mode == "outside":
        allowed = tmp_path / "unrelated";allowed.mkdir();spec["allowed_roots"] = [str(allowed)]
    elif mode == "file":
        file = next((root / "data").rglob("*.parquet"));target = tmp_path / "copied.parquet";target.write_bytes(file.read_bytes());file.unlink();file.symlink_to(target)
    else:
        directory = next((root / "data").iterdir());directory.rename(root / "moved");directory.symlink_to(root / "moved", target_is_directory=True)
    with pytest.raises(ValueError, match="DATASET_TIMESTAMPS_INVALID"):
        check.inspect_dataset_timestamps(data, spec)


@pytest.mark.parametrize("mode", ["missing-column", "nan", "episode-mismatch", "float-episode", "nested-timestamp"])
def test_real_column_schema_and_values_are_hard_requirements(tmp_path, parquet, mode):
    data, spec = dataset(tmp_path, [(0, [0, .05, .1])])
    file = next((Path(data["path"]) / "data").rglob("*.parquet"))
    extras = ({"columns": ["timestamp"]} if mode == "missing-column" else
              {"episode_dtype": "float32"} if mode == "float-episode" else
              {"timestamp_type": "list<item: double>"} if mode == "nested-timestamp" else {})
    _fake_write(file, [0, float("nan"), .1] if mode == "nan" else [0, .05, .1], [0, 1, 0] if mode == "episode-mismatch" else [0, 0, 0], **extras)
    _update_manifest(data, spec)
    with pytest.raises(ValueError, match="DATASET_TIMESTAMPS_INVALID"):
        check.inspect_dataset_timestamps(data, spec)


def test_missing_runtime_or_optional_reader_is_an_explicit_blocker(tmp_path, monkeypatch):
    data, spec = dataset(tmp_path)
    original = check.importlib.import_module
    monkeypatch.setattr(check.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("missing")) if name == check.MODULE else original(name))
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_CHECK_UNAVAILABLE"):
        check.inspect_dataset_timestamps(data, spec)


def test_runtime_tolerance_is_read_from_the_actual_constructor(runtime, monkeypatch):
    receipt, tolerance = check._runtime()
    assert tolerance == 1e-4 and receipt["module_sha256"] == _hash(Path(runtime.__file__))
    monkeypatch.setattr(runtime.LeRobotDataset, "__init__", lambda self, tolerance_s=None: None)
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_CHECK_UNAVAILABLE"):
        check._runtime()


def test_modified_runtime_formula_cannot_receive_a_verified_receipt(runtime, monkeypatch):
    monkeypatch.setattr(runtime, "check_timestamps_sync", lambda *args: True)
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_CHECK_UNAVAILABLE"):
        check._runtime()


def test_mutation_during_streaming_is_detected(tmp_path, parquet, monkeypatch):
    data, spec = dataset(tmp_path, [(0, [0, .05, .1])])
    file = next((Path(data["path"]) / "data").rglob("*.parquet"))
    original = parquet.ParquetFile.iter_batches
    def changed(self, **kwargs):
        yield from original(self, **kwargs)
        file.write_bytes(file.read_bytes() + b"x")
    monkeypatch.setattr(parquet.ParquetFile, "iter_batches", changed)
    with pytest.raises(ValueError, match="DATASET_TIMESTAMP_SOURCE_CHANGED"):
        check.inspect_dataset_timestamps(data, spec)


def _arrow_writer(path, timestamps, indices):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    parquet.write_table(arrow.table({"timestamp": arrow.array(timestamps, type=arrow.float32()),
                                     "episode_index": arrow.array(indices, type=arrow.int64()),
                                     "unused_video": ["not decoded"] * len(timestamps)}), path)


def test_real_parquet_full_scan_has_deterministic_read_only_evidence(tmp_path, runtime):
    pytest.importorskip("pyarrow")
    data, spec = dataset(tmp_path, [(0, [i / 20 for i in range(10000)])], writer=_arrow_writer)
    before = {str(path): _hash(path) for path in Path(data["path"]).rglob("*") if path.is_file()}
    result = check.inspect_dataset_timestamps(data, spec)
    assert result["checked_rows"] == 10000 and result["checked_pairs"] == 9999
    assert before == {str(path): _hash(path) for path in Path(data["path"]).rglob("*") if path.is_file()}


def test_real_parquet_checks_late_rows_not_only_a_sample(tmp_path, runtime):
    pytest.importorskip("pyarrow")
    timestamps = [i / 20 for i in range(10000)]
    timestamps[-1] += .1
    data, spec = dataset(tmp_path, [(0, timestamps)], writer=_arrow_writer)
    with pytest.raises(ValueError, match="DATASET_TIMESTAMPS_INVALID"):
        check.inspect_dataset_timestamps(data, spec)


def test_real_parquet_timestamp_conflict_is_the_reported_20_vs_60_failure(tmp_path, runtime):
    pytest.importorskip("pyarrow")
    data, spec = dataset(tmp_path, [(0, [0, 1 / 60, 2 / 60])], writer=_arrow_writer)
    with pytest.raises(ValueError, match="期望帧间隔 0.05 秒"):
        check.inspect_dataset_timestamps(data, spec)
