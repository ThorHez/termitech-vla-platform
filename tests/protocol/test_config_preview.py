import hashlib

import pytest

from vla_protocol.config_preview import MAX_CONFIG_SOURCE_BYTES, validate_generated_config


def evidence(source):
    return {"schema_version": 1, "language": "python", "filename": "training_config.py",
            "source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}


def test_python_configuration_is_checked_without_execution(tmp_path):
    target = tmp_path / "never-written"
    value = evidence(f"from pathlib import Path\nPath({str(target)!r}).write_text('not executed')\n")
    assert validate_generated_config(value) is value
    assert not target.exists()


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"language": "javascript"}, {"filename": "../training_config.py"},
    {"source": "changed = True\n"}, {"sha256": "x" * 64}, {"source": None},
])
def test_invalid_or_tampered_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        validate_generated_config({**evidence("config = {}\n"), **change})


@pytest.mark.parametrize("source", ["", "config = (\n", "# " + "字" * (MAX_CONFIG_SOURCE_BYTES // 3 + 1)])
def test_empty_invalid_and_oversized_sources_are_rejected(source):
    with pytest.raises(ValueError):
        validate_generated_config(evidence(source))
