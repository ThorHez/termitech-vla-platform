"""Catalog defaults must preserve the selected class's own base configuration."""

import importlib.util
import sys

import pytest

from vla_adapters.data_processors import catalog_data_processors, instantiate_data_processor


SOURCE = '''import dataclasses
CALLS = []
@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None
@dataclasses.dataclass(frozen=True)
class DataConfig:
    prompt_from_task: bool = False
    use_quantile_norm: bool = False
    action_sequence_keys: tuple[str, ...] = ('actions',)
@dataclasses.dataclass(frozen=True)
class DataConfigFactory:
    repo_id: str = ''
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: DataConfig | None = None
def configured_base():
    CALLS.append('base-created')
    return DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=('custom_actions',))
@dataclasses.dataclass(frozen=True)
class ConstructedBaseFactory(DataConfigFactory):
    base_config: DataConfig = dataclasses.field(default_factory=configured_base)
@dataclasses.dataclass(frozen=True)
class ExplicitBaseFactory(DataConfigFactory):
    base_config: DataConfig = DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=('custom_actions',))
@dataclasses.dataclass(frozen=True, kw_only=True)
class RequiredBaseFactory(DataConfigFactory):
    base_config: DataConfig = dataclasses.field()
'''


@pytest.fixture
def catalog_module(tmp_path, monkeypatch):
    path = tmp_path / "processor_defaults.py"
    path.write_text(SOURCE)
    spec = importlib.util.spec_from_file_location("processor_default_preservation_fixture", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, tmp_path


def descriptor(module, root, name):
    return next(item for item in catalog_data_processors(module, root) if item["name"] == name)


def defaults_from_catalog(entry):
    # Mirrors the UI's processorDefaults() without introducing a JS dependency.
    return {item["path"]: item["default"] for item in entry["parameters"] if "default" in item}


def test_unknown_base_default_factory_is_not_called_or_replaced_by_type_defaults(catalog_module):
    module, root = catalog_module
    entry = descriptor(module, root, "ConstructedBaseFactory")
    assert module.CALLS == []
    nested = [item for item in entry["parameters"] if item["path"].startswith("base_config.")]
    assert nested and all("default" not in item for item in nested)
    factory = instantiate_data_processor({"name": entry["name"], "parameters": defaults_from_catalog(entry)}, module, root)
    assert factory.base_config.prompt_from_task and factory.base_config.use_quantile_norm
    assert factory.base_config.action_sequence_keys == ("custom_actions",)
    assert module.CALLS == ["base-created"]


def test_concrete_base_config_defaults_describe_the_instance_not_the_type(catalog_module):
    module, root = catalog_module
    entry = descriptor(module, root, "ExplicitBaseFactory")
    values = defaults_from_catalog(entry)
    assert values["base_config.prompt_from_task"] is True
    assert values["base_config.use_quantile_norm"] is True
    assert values["base_config.action_sequence_keys"] == ["custom_actions"]
    factory = instantiate_data_processor({"name": entry["name"], "parameters": values}, module, root)
    assert factory.base_config == module.ExplicitBaseFactory().base_config
    assert module.CALLS == []


def test_required_base_config_is_not_advertised_as_constructible_without_a_value(catalog_module):
    module, root = catalog_module
    entry = descriptor(module, root, "RequiredBaseFactory")
    assert not entry["available"]
    assert any(item["path"] == "base_config" and item["required"] for item in entry["unsupported_parameters"])
