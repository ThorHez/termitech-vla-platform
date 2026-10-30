from types import SimpleNamespace

import psutil
import pytest

from vla_agent import metrics


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(metrics.psutil, "cpu_count", lambda logical=True: 64)
    monkeypatch.setattr(metrics.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: [0, 2, 4, 6]))


def test_logical_and_current_process_affinity_are_reported(cpu):
    errors = {}
    assert metrics._cpu_capacity(errors) == {"cpu_count": 64, "available_cpu_count": 4}
    assert errors == {}


def test_single_affinity_cpu_is_not_rounded_to_zero(cpu, monkeypatch):
    monkeypatch.setattr(metrics.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: [7]))
    assert metrics._cpu_capacity({})["available_cpu_count"] == 1


def test_platform_without_affinity_uses_logical_count(cpu, monkeypatch):
    monkeypatch.setattr(metrics.psutil, "Process", lambda: SimpleNamespace())
    errors = {}
    assert metrics._cpu_capacity(errors) == {"cpu_count": 64, "available_cpu_count": 64}
    assert errors == {}


@pytest.mark.parametrize("error", [psutil.AccessDenied(), OSError("unavailable"), NotImplementedError()])
def test_affinity_read_failure_falls_back_without_breaking_metrics(cpu, monkeypatch, error):
    def fail():
        raise error
    monkeypatch.setattr(metrics.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=fail))
    errors = {}
    assert metrics._cpu_capacity(errors)["available_cpu_count"] == 64
    assert errors == {"cpu_affinity": type(error).__name__}


def test_missing_logical_count_can_still_use_affinity(cpu, monkeypatch):
    monkeypatch.setattr(metrics.psutil, "cpu_count", lambda logical=True: None)
    errors = {}
    assert metrics._cpu_capacity(errors) == {"cpu_count": None, "available_cpu_count": 4}
    assert errors == {"cpu_count": "Unavailable"}


def test_all_counts_unknown_are_null_with_collection_errors(cpu, monkeypatch):
    monkeypatch.setattr(metrics.psutil, "cpu_count", lambda logical=True: None)
    monkeypatch.setattr(metrics.psutil, "Process", lambda: SimpleNamespace())
    errors = {}
    assert metrics._cpu_capacity(errors) == {"cpu_count": None, "available_cpu_count": None}
    assert errors == {"cpu_count": "Unavailable", "available_cpu_count": "Unavailable"}


def test_cpu_count_errors_do_not_break_other_system_metrics(cpu, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("cpu unavailable")
    monkeypatch.setattr(metrics.psutil, "cpu_count", fail)
    monkeypatch.setattr(metrics.psutil, "Process", fail)
    result = metrics.system_metrics(str(tmp_path))
    assert result["cpu_count"] is None and result["available_cpu_count"] is None
    assert result["disk_total_bytes"] > 0
    assert result["memory_total_bytes"] > 0
    assert result["collection_errors"]["cpu_count"] == "OSError"


def test_empty_affinity_falls_back_and_does_not_report_zero(cpu, monkeypatch):
    monkeypatch.setattr(metrics.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: []))
    errors = {}
    assert metrics._cpu_capacity(errors)["available_cpu_count"] == 64
    assert errors == {"cpu_affinity": "Unavailable"}
