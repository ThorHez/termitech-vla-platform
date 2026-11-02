from types import ModuleType
import sys

import pytest

from vla_adapters import dataset_source_validation as source

np = pytest.importorskip("numpy")


@pytest.fixture
def raw_dataset(monkeypatch):
    state = {"indices": [], "invalid": False}

    class Dataset:
        def __init__(self, path):
            state["path"] = path

        def __len__(self):
            return 7

        def __getitem__(self, index):
            state["indices"].append(index)
            return {"state": np.array([0.0, float("nan") if state["invalid"] else 1.0]),
                    "actions": np.zeros((16, 10)), "timestamp": np.array(index / 20),
                    "image": np.zeros((3, 4, 5), np.float32), "task": "private demonstration text"}

    module = ModuleType("lerobot.common.datasets.lerobot_dataset")
    module.LeRobotDataset = Dataset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    info = {"total_frames": 7, "features": {
        "state": {"dtype": "float32", "shape": [2]},
        "actions": {"dtype": "float32", "shape": [16, 10]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "image": {"dtype": "image", "shape": [4, 5, 3]},
        "task": {"dtype": "string", "shape": [1]},
    }}
    return state, info


def test_registration_decodes_only_first_middle_last_and_returns_no_data_values(raw_dataset, tmp_path):
    state, info = raw_dataset
    result = source._raw_samples(tmp_path, info)
    assert state["indices"] == [0, 3, 6]
    assert result["status"] == "verified"
    assert result["raw_samples"]["checked_samples"] == 3
    fields = result["raw_samples"]["samples"][0]["fields"]
    assert fields["image"] == {"shape": [3, 4, 5], "dtype": "float32"}
    assert fields["actions"]["shape"] == [16, 10]
    assert "task" not in fields
    assert "private demonstration" not in str(result)
    assert all(set(value) == {"shape", "dtype"} for value in fields.values())


def test_registration_rejects_nonfinite_raw_state(raw_dataset, tmp_path):
    state, info = raw_dataset
    state["invalid"] = True
    with pytest.raises(ValueError, match="DATASET_SOURCE_INVALID"):
        source._raw_samples(tmp_path, info)


def test_registration_rejects_actual_field_shape_mismatch(raw_dataset, tmp_path):
    _, info = raw_dataset
    info["features"]["actions"]["shape"] = [50, 10]
    with pytest.raises(ValueError, match="声明形状"):
        source._raw_samples(tmp_path, info)


def test_registration_rejects_loader_frame_count_mismatch(raw_dataset, tmp_path):
    _, info = raw_dataset
    info["total_frames"] = 8
    with pytest.raises(ValueError, match="帧数"):
        source._raw_samples(tmp_path, info)
