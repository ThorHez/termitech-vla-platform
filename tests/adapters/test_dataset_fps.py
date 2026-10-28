"""Registered FPS policy and runtime isolation; no Torch, GPU or training steps."""

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import ModuleType

import pytest

from vla_adapters import dataset_fps


FAKE_RUNTIME = '''import json
from pathlib import Path
TIMESTAMP_CHECKS = []
def load_info(root):
    return json.loads((Path(root) / "meta/info.json").read_text())
class LeRobotDatasetMetadata:
    def __init__(self, repo_id, root=None, revision=None, force_cache_sync=False):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else Path(repo_id)
        self.load_metadata()
        self.original_info = self.info
    def load_metadata(self):
        self.info = load_info(self.root)
    @property
    def fps(self):
        """The original metadata time base."""
        return self.info["fps"]
class LeRobotDataset:
    def __init__(self, repo_id, delta_timestamps=None, root=None):
        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.timestamps = self.meta.info.get("timestamps", [])
        TIMESTAMP_CHECKS.append(self.fps)
        for previous, current in zip(self.timestamps, self.timestamps[1:]):
            if abs((current - previous) - 1 / self.fps) > 1e-4:
                raise ValueError("REAL_TIMESTAMP_SYNC_FAILURE")
        self.delta_indices = {key: [round(t * self.fps) for t in values]
                              for key, values in (delta_timestamps or {}).items()}
    @property
    def fps(self):
        return self.meta.fps
'''


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for name in ("lerobot", "lerobot.common", "lerobot.common.datasets"):
        package = ModuleType(name);package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    path = tmp_path / "lerobot_dataset.py";path.write_text(FAKE_RUNTIME)
    loader = importlib.util.spec_from_file_location(dataset_fps.MODULE, path)
    module = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, dataset_fps.MODULE, module)
    loader.loader.exec_module(module)
    original = module.LeRobotDatasetMetadata.__init__
    try:
        yield module
    finally:
        module.LeRobotDatasetMetadata.__init__ = original
        if hasattr(module.LeRobotDatasetMetadata, dataset_fps.HOOK):
            delattr(module.LeRobotDatasetMetadata, dataset_fps.HOOK)


def asset(tmp_path, name, *, registered=20, metadata=60, timestamps=None):
    path = tmp_path / name
    (path / "meta").mkdir(parents=True)
    document = {"fps": metadata, "features": {"camera": {"info": {"video.fps": metadata}}}}
    if timestamps is not None:
        document["timestamps"] = timestamps
    (path / "meta/info.json").write_text(json.dumps(document))
    return {"id": name, "path": str(path), "dataset_profile": {"fps": registered}}


def single(row, root):
    return {"inputs": {"dataset": row}, "allowed_roots": [str(root)]}


def mixture(rows, root):
    return {"inputs": {"dataset_" + str(index): row for index, row in enumerate(rows)},
            "dataset_bindings": [{"dataset_asset_id": row["id"]} for row in rows],
            "allowed_roots": [str(root)]}


def test_policy_is_frozen_per_bound_role_and_uses_only_stdlib(tmp_path, monkeypatch):
    row = asset(tmp_path, "first", registered=20, metadata=60)
    spec = single(row, tmp_path)
    spec["inputs"]["dataset_99"] = {"path": "/not-selected", "dataset_profile": {"fps": 99}}
    before = deepcopy(spec)
    monkeypatch.setattr(dataset_fps.importlib, "import_module", lambda *_: pytest.fail("Resolution must not import LeRobot"))
    policy = dataset_fps.resolve_registered_fps(spec)
    digest = hashlib.sha256((Path(row["path"]) / "meta/info.json").read_bytes()).hexdigest()
    assert policy == [{"role": "dataset", "path": row["path"], "registered_fps": 20,
                       "metadata_fps": 60, "effective_fps": 20, "source": "registered", "info_sha256": digest}]
    policy[0]["effective_fps"] = 99
    assert spec == before
    assert dataset_fps.resolve_registered_fps(spec)[0]["effective_fps"] == 20


def test_mixture_none_registration_does_not_borrow_another_roles_override(tmp_path):
    rows = [asset(tmp_path, "one", registered=20, metadata=60),
            asset(tmp_path, "two", registered=None, metadata=30)]
    policy = dataset_fps.resolve_registered_fps(mixture(rows, tmp_path))
    assert [(row["role"], row["effective_fps"], row["source"]) for row in policy] == [
        ("dataset_0", 20, "registered"), ("dataset_1", 30, "metadata")]


def test_resolver_honors_only_actual_binding_order_and_identities(tmp_path):
    first, second = asset(tmp_path, "first"), asset(tmp_path, "second", registered=30)
    spec = mixture([second, first], tmp_path)
    assert [row["path"] for row in dataset_fps.resolve_registered_fps(spec)] == [second["path"], first["path"]]
    spec["inputs"]["dataset_1"]["id"] = "wrong-id"
    with pytest.raises(ValueError, match="DATASET_FPS_INVALID"):
        dataset_fps.resolve_registered_fps(spec)


@pytest.mark.parametrize("registered", [0, -1, True, "20", float("nan"), float("inf"), 1001, 10**400])
def test_invalid_registration_remains_a_hard_error(tmp_path, registered):
    with pytest.raises(ValueError, match="DATASET_FPS_INVALID"):
        dataset_fps.resolve_registered_fps(single(asset(tmp_path, "one", registered=registered), tmp_path))


@pytest.mark.parametrize("metadata", [0, -1, True, None, "60", float("nan"), float("inf")])
def test_registered_policy_requires_valid_metadata_comparison(tmp_path, metadata):
    with pytest.raises(ValueError, match="DATASET_FPS_INVALID"):
        dataset_fps.resolve_registered_fps(single(asset(tmp_path, "one", metadata=metadata), tmp_path))


def test_legacy_missing_fps_does_not_import_lerobot_or_add_a_new_block(tmp_path, monkeypatch):
    row = asset(tmp_path, "one", registered=None, metadata=None)
    path = Path(row["path"]) / "meta/info.json"
    path.write_text('{"features":{}}')
    monkeypatch.delitem(sys.modules, dataset_fps.MODULE, raising=False)
    monkeypatch.setattr(dataset_fps.importlib, "import_module", lambda *_: pytest.fail("No declaration requires no runtime import"))
    receipt = dataset_fps.install_registered_fps(single(row, tmp_path))
    assert receipt["runtime"] is None and receipt["installed"] is False
    assert receipt["policy"][0]["effective_fps"] is None


@pytest.mark.parametrize("other_registered", [30, None])
def test_same_canonical_path_cannot_take_conflicting_effective_policies(tmp_path, other_registered):
    row = asset(tmp_path, "one", registered=20, metadata=60)
    alias = tmp_path / "alias";alias.symlink_to(Path(row["path"]), target_is_directory=True)
    other = {**deepcopy(row), "id": "alias", "path": str(alias), "dataset_profile": {"fps": other_registered}}
    with pytest.raises(ValueError, match="DATASET_FPS_POLICY_CONFLICT"):
        dataset_fps.resolve_registered_fps(mixture([row, other], tmp_path))


def test_duplicate_canonical_path_with_equal_effective_fps_remains_supported(tmp_path):
    row = asset(tmp_path, "one", registered=20, metadata=20)
    other = {**deepcopy(row), "id": "alias", "dataset_profile": {"fps": None}}
    policy = dataset_fps.resolve_registered_fps(mixture([row, other], tmp_path))
    assert len(policy) == 2 and policy[0]["path"] == policy[1]["path"]
    assert policy[1]["registered_fps"] is None and policy[1]["source"] == "metadata"


def test_managed_scope_is_checked_before_reading_metadata(tmp_path):
    row = asset(tmp_path, "one")
    allowed = tmp_path / "allowed";allowed.mkdir()
    with pytest.raises(ValueError, match="DATASET_FPS_INVALID"):
        dataset_fps.resolve_registered_fps(single(row, allowed))


def test_both_metadata_instances_and_existing_class_aliases_use_effective_fps_without_writing(tmp_path, runtime):
    row = asset(tmp_path, "one", timestamps=[0, .05, .1])
    spec, alias = single(row, tmp_path), runtime.LeRobotDatasetMetadata
    info_path = Path(row["path"]) / "meta/info.json";before = info_path.read_bytes()
    original = alias.__init__
    with dataset_fps.registered_fps_context(spec) as receipt:
        assert runtime.LeRobotDatasetMetadata is alias and alias.__init__ is not original
        external = alias(row["path"])
        dataset = runtime.LeRobotDataset(row["path"], {"actions": [index / external.fps for index in range(4)]})
        assert external.fps == dataset.fps == 20
        assert dataset.delta_indices == {"actions": [0, 1, 2, 3]}
        assert external.info is not external.original_info
        assert external.original_info["fps"] == dataset.meta.original_info["fps"] == 60
        assert dataset.meta.info["features"]["camera"]["info"]["video.fps"] == 60
        assert dataset.timestamps == [0, .05, .1]
        assert receipt["runtime"]["timestamp_validation"] == "unchanged"
        assert receipt["runtime"]["resampling"] is False and receipt["runtime"]["writes_metadata"] is False
        assert receipt["runtime"]["module_sha256"] == hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    assert alias.__init__ is original and not hasattr(alias, dataset_fps.HOOK)
    assert info_path.read_bytes() == before


def test_mixed_fps_targets_are_instance_scoped_and_other_datasets_stay_unchanged(tmp_path, runtime):
    one = asset(tmp_path, "one", registered=20, metadata=60, timestamps=[0, .05, .1])
    two = asset(tmp_path, "two", registered=30, metadata=50, timestamps=[0, 1/30, 2/30])
    untouched = asset(tmp_path, "unbound", registered=None, metadata=60)
    with dataset_fps.registered_fps_context(mixture([one, two], tmp_path)):
        for row, fps in ((one, 20), (two, 30)):
            external = runtime.LeRobotDatasetMetadata(row["path"])
            dataset = runtime.LeRobotDataset(row["path"], {"actions": [index / external.fps for index in range(3)]})
            assert external.fps == dataset.fps == fps
            assert dataset.delta_indices["actions"] == [0, 1, 2]
        assert runtime.LeRobotDatasetMetadata(untouched["path"]).fps == 60


def test_unregistered_selected_dataset_keeps_its_own_metadata_fps(tmp_path, runtime):
    rows = [asset(tmp_path, "one", registered=20, metadata=60), asset(tmp_path, "two", registered=None, metadata=30)]
    with dataset_fps.registered_fps_context(mixture(rows, tmp_path)):
        assert runtime.LeRobotDatasetMetadata(rows[0]["path"]).fps == 20
        assert runtime.LeRobotDatasetMetadata(rows[1]["path"]).fps == 30


def test_real_timestamp_mismatch_is_not_suppressed_and_context_restores_after_failure(tmp_path, runtime):
    row = asset(tmp_path, "one", timestamps=[0, 1/60, 2/60])
    original = runtime.LeRobotDatasetMetadata.__init__
    with pytest.raises(ValueError, match="REAL_TIMESTAMP_SYNC_FAILURE"):
        with dataset_fps.registered_fps_context(single(row, tmp_path)):
            runtime.LeRobotDataset(row["path"])
    assert runtime.TIMESTAMP_CHECKS == [20]
    assert runtime.LeRobotDatasetMetadata.__init__ is original
    assert runtime.LeRobotDatasetMetadata(row["path"]).fps == 60


def test_ordinary_exported_builder_installs_fps_that_survives_its_return(tmp_path, runtime):
    row = asset(tmp_path, "one")
    spec = single(row, tmp_path)

    def build_config():
        return {"repo_id": row["path"], "fps_receipt": dataset_fps.install_registered_fps(spec)}

    config = build_config()
    assert runtime.LeRobotDatasetMetadata(config["repo_id"]).fps == 20
    assert config["fps_receipt"] == dataset_fps.install_registered_fps(spec)


def test_idempotence_and_nested_contexts_do_not_remove_outer_ownership(tmp_path, runtime):
    row = asset(tmp_path, "one");spec = single(row, tmp_path)
    original = runtime.LeRobotDatasetMetadata.__init__
    with dataset_fps.registered_fps_context(spec) as receipt:
        wrapper = runtime.LeRobotDatasetMetadata.__init__
        assert dataset_fps.install_registered_fps(spec) == receipt
        with dataset_fps.registered_fps_context(spec) as inner:
            assert inner == receipt and runtime.LeRobotDatasetMetadata.__init__ is wrapper
        assert runtime.LeRobotDatasetMetadata.__init__ is wrapper
    assert runtime.LeRobotDatasetMetadata.__init__ is original


@pytest.mark.parametrize("registered", [30, None])
def test_conflicting_installation_cannot_replace_an_existing_policy(tmp_path, runtime, registered):
    row = asset(tmp_path, "one");spec = single(row, tmp_path)
    receipt = dataset_fps.install_registered_fps(spec)
    wrapper = runtime.LeRobotDatasetMetadata.__init__
    changed = deepcopy(spec);changed["inputs"]["dataset"]["dataset_profile"]["fps"] = registered
    with pytest.raises(ValueError, match="DATASET_FPS_POLICY_CONFLICT"):
        dataset_fps.install_registered_fps(changed)
    assert runtime.LeRobotDatasetMetadata.__init__ is wrapper
    assert runtime.LeRobotDatasetMetadata(row["path"]).fps == 20
    assert dataset_fps.install_registered_fps(spec) == receipt


def test_mutating_returned_receipt_or_input_spec_cannot_change_the_installed_map(tmp_path, runtime):
    row = asset(tmp_path, "one");spec = single(row, tmp_path)
    receipt = dataset_fps.install_registered_fps(spec)
    receipt["policy"][0]["effective_fps"] = 99
    spec["inputs"]["dataset"]["dataset_profile"]["fps"] = 33
    spec["allowed_roots"][:] = ["/"]
    assert runtime.LeRobotDatasetMetadata(row["path"]).fps == 20


def test_metadata_file_drift_is_rejected_instead_of_overriding_unreviewed_source(tmp_path, runtime):
    row = asset(tmp_path, "one")
    with dataset_fps.registered_fps_context(single(row, tmp_path)):
        path = Path(row["path"]) / "meta/info.json"
        document = json.loads(path.read_text());document["fps"] = 50
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="DATASET_FPS_METADATA_CHANGED"):
            runtime.LeRobotDatasetMetadata(row["path"])


def test_spawn_style_pickle_preserves_effective_info_and_delta_indices_after_hook_restoration(tmp_path, runtime):
    row = asset(tmp_path, "one", timestamps=[0, .05, .1])
    with dataset_fps.registered_fps_context(single(row, tmp_path)):
        dataset = runtime.LeRobotDataset(row["path"], {"actions": [0, .05, .1]})
        serialized = pickle.dumps(dataset)
    restored = pickle.loads(serialized)
    assert restored.fps == 20 and restored.delta_indices["actions"] == [0, 1, 2]
    assert runtime.LeRobotDatasetMetadata(row["path"]).fps == 60


def test_fractional_fps_is_not_truncated(tmp_path, runtime):
    row = asset(tmp_path, "one", registered=29.97, metadata=60)
    with dataset_fps.registered_fps_context(single(row, tmp_path)) as receipt:
        assert runtime.LeRobotDatasetMetadata(row["path"]).fps == 29.97
        assert receipt["policy"][0]["effective_fps"] == 29.97


@pytest.mark.parametrize("change", ["signature", "fps_property", "loader", "source"])
def test_unknown_runtime_semantics_fail_closed_without_installing(tmp_path, runtime, monkeypatch, change):
    row = asset(tmp_path, "one")
    cls = runtime.LeRobotDatasetMetadata
    if change == "signature":
        monkeypatch.setattr(cls, "__init__", lambda self, repo_id, **kwargs: None)
    elif change == "fps_property":
        monkeypatch.setattr(cls, "fps", property(lambda self: self.info.get("fps", 30)))
    elif change == "loader":
        monkeypatch.setattr(cls, "load_metadata", lambda self: None)
    else:
        monkeypatch.setattr(runtime, "__file__", str(tmp_path / "other.py"))
        Path(runtime.__file__).write_text("# unrelated dynamic source\n")
    original = cls.__init__
    with pytest.raises(ValueError, match="DATASET_FPS_RUNTIME_UNSUPPORTED"):
        dataset_fps.install_registered_fps(single(row, tmp_path))
    assert cls.__init__ is original and not hasattr(cls, dataset_fps.HOOK)


def test_compatibility_evidence_accepts_the_real_installed_lerobot_source_without_importing_torch(tmp_path, runtime, monkeypatch):
    # This file is byte-identical to the LeRobot implementation inspected on 222.
    # Compile its three relevant methods only; their source locations/signatures
    # remain real, while unrelated Torch/HF imports and disk loading do not run.
    source = Path(__file__).resolve().parents[2] / 'third_party/openpi-umi/third_party/lerobot/lerobot/common/datasets/lerobot_dataset.py'
    tree = ast.parse(source.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LeRobotDatasetMetadata')
    cls = ast.ClassDef(name=original.name, bases=[], keywords=[], decorator_list=[], body=[
        node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in {'__init__', 'load_metadata', 'fps'}])
    selected = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(selected)
    namespace = {'__name__': dataset_fps.MODULE, 'Path': Path}
    exec(compile(selected, str(source), 'exec'), namespace)
    monkeypatch.setattr(runtime, 'LeRobotDatasetMetadata', namespace['LeRobotDatasetMetadata'])
    monkeypatch.setattr(runtime, '__file__', str(source))
    with dataset_fps.registered_fps_context(single(asset(tmp_path, 'one'), tmp_path)) as receipt:
        assert receipt['runtime']['module_path'] == str(source.resolve())
        assert receipt['runtime']['module_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert 'force_cache_sync' in receipt['runtime']['constructor_signature']


def test_existing_foreign_hook_is_not_silently_overwritten_or_removed(tmp_path, runtime):
    spec = single(asset(tmp_path, 'one'), tmp_path)
    foreign = lambda self, *args, **kwargs: None
    with pytest.raises(ValueError, match='DATASET_FPS_POLICY_CONFLICT'):
        with dataset_fps.registered_fps_context(spec):
            runtime.LeRobotDatasetMetadata.__init__ = foreign
            with pytest.raises(ValueError, match='DATASET_FPS_POLICY_CONFLICT'):
                dataset_fps.install_registered_fps(spec)
    assert runtime.LeRobotDatasetMetadata.__init__ is foreign
