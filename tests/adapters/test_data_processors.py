import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vla_adapters import data_processors
from vla_adapters.data_processors import (catalog_data_processors, instantiate_data_processor,
                                         normalize_processor, validate_processor_instance)
from vla_adapters.dataset_bindings import normalize_dataset_bindings
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.training_plan import fingerprint, inspect_training_config

from test_multi_dataset_adapters import explicit
from test_training_plan import training  # noqa: F401


SOURCE = '''from __future__ import annotations
import abc, dataclasses
from collections.abc import Sequence
from typing import Literal
CALLS=[]
def mapping_default():
    CALLS.append('mapping-default')
    return {'images':'observation.camera'}
@dataclasses.dataclass
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None
@dataclasses.dataclass
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: object = None
    prompt_from_task: bool = False
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ('actions',)
    action_loss_mask: Sequence[float] | None = None
    robot_type: str | None = None
@dataclasses.dataclass
class DataConfigFactory(abc.ABC):
    repo_id: str = ''
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: DataConfig | None = None
    @abc.abstractmethod
    def create(self): pass
@dataclasses.dataclass
class IndependentFactory(DataConfigFactory):
    """A processor exported independently of every named recipe."""
    image_horizon: int = 2
    training_mode: bool = True
    temperature: float = 0.1
    action_sequence_keys: Sequence[str] = ('actions',)
    default_prompt: str | None = None
    mode: Literal['relative', 'absolute'] = 'relative'
    mapping: dict[str,str] = dataclasses.field(default_factory=mapping_default)
    def create(self): return self.base_config
@dataclasses.dataclass(kw_only=True)
class RequiredFactory(DataConfigFactory):
    count: int
    def create(self): return self.base_config
@dataclasses.dataclass(kw_only=True)
class RequiredComplexFactory(DataConfigFactory):
    transform: object
    def create(self): return self.base_config
@dataclasses.dataclass(kw_only=True)
class RequiredACPFactory(IndependentFactory):
    acp_dropout: float
    mapping: dict[str,str] = dataclasses.field(default_factory=lambda: {'actions':'actions','is_positive':'is_positive'})
@dataclasses.dataclass
class FakeDataConfig(DataConfigFactory):
    def create(self): return 'fake'
@dataclasses.dataclass
class MultiDataConfigFactory(DataConfigFactory):
    datasets: list[DataConfigFactory] = dataclasses.field(default_factory=list)
    def create(self): return 'multi'
@dataclasses.dataclass
class RLDSDroidDataConfig(DataConfigFactory):
    rlds_data_dir: str | None = None
    def create(self): return 'rlds'
Alias = IndependentFactory
'''


@pytest.fixture
def processor_module(training, monkeypatch):
    _, spec, _, _ = training
    source = Path(spec['code_dir']) / 'processor_config.py'
    source.write_text(SOURCE)
    module_spec = importlib.util.spec_from_file_location('registered_processor_config', source)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, module)
    module_spec.loader.exec_module(module)
    return module


def processor(parameters=None, name='IndependentFactory'):
    return {'name': name, 'parameters': parameters or {}}


def test_catalog_inspects_typed_defaults_without_instantiating_factories(processor_module, training):
    module = processor_module
    entries = {row['name']: row for row in catalog_data_processors(module, training[1]['code_dir'])}
    assert module.CALLS == []
    assert 'Alias' not in entries
    params = {row['path']: row for row in entries['IndependentFactory']['parameters']}
    assert entries['IndependentFactory']['available'] and entries['RequiredFactory']['available']
    assert params['image_horizon'] == {'path': 'image_horizon', 'type': 'integer', 'required': False, 'default': 2}
    assert params['default_prompt']['type'] == 'string' and params['default_prompt']['nullable']
    assert params['action_sequence_keys']['type'] == 'array' and params['action_sequence_keys']['default'] == ['actions']
    assert params['mode']['choices'] == ['relative', 'absolute']
    assert params['base_config.prompt_from_task']['type'] == 'boolean'
    assert 'mapping' in params and 'default' not in params['mapping']
    assert not {'repo_id', 'assets', 'base_config.repo_id', 'base_config.norm_stats'} & set(params)
    assert not entries['RequiredComplexFactory']['available']
    assert entries['RequiredComplexFactory']['unsupported_parameters'][0]['path'] == 'transform'
    assert all(not entries[name]['available'] for name in ['FakeDataConfig', 'MultiDataConfigFactory', 'RLDSDroidDataConfig'])
    assert all(row['source']['path'] == 'processor_config.py' for row in entries.values())


@pytest.mark.parametrize('invalid', [None, {}, {'name': 'a.b', 'parameters': {}}, {'name': '__import__', 'parameters': {}},
                                    {'name': 'Factory()', 'parameters': {}}, {'name': 'Factory', 'parameters': {'image_horizon': float('nan')}},
                                    {'name': 'Factory', 'parameters': {'__class__': 'x'}},
                                    {'name': 'Factory', 'parameters': {'a': [0] * 65}}])
def test_processor_contract_rejects_code_names_and_unbounded_literals(invalid):
    with pytest.raises(ValueError, match='DATA_PROCESSOR_INVALID'):
        normalize_processor(invalid)


@pytest.mark.parametrize('parameters', [{'repo_id': '/data'}, {'assets.assets_dir': '/norm'}, {'base_config.norm_stats': {}},
                                       {'weights': [1]}, {'base_config': {'repo_id': '/hidden'}}])
def test_processor_parameters_cannot_override_assets(parameters):
    with pytest.raises(ValueError, match='DATA_PROCESSOR_ASSET_FIELDS_MANAGED'):
        normalize_processor(processor(parameters))


def test_binding_shapes_keep_legacy_digests_and_do_not_alias_processor_parameters():
    original = {'dataset_asset_id': 'd', 'norm_asset_id': 'n', 'recipe_index': 1, 'weight': 2.0}
    assert normalize_dataset_bindings([original]) == [original]
    assert normalize_dataset_bindings([{**original, 'processor': None}]) == [original]
    modern = {'dataset_asset_id': 'd', 'norm_asset_id': 'n', 'processor': processor({'image_horizon': 2}), 'weight': 2.0}
    assert normalize_dataset_bindings([{**modern, 'recipe_index': None}]) == [modern]
    copy = normalize_dataset_bindings([modern])
    copy[0]['processor']['parameters']['image_horizon'] = 99
    assert modern['processor']['parameters']['image_horizon'] == 2
    for bad in [{**original, 'processor': processor()}, {**original, 'recipe_index': None}, {**modern, 'processor': None}]:
        with pytest.raises(ValueError, match='DATASET_BINDINGS_INVALID'):
            normalize_dataset_bindings([bad])


def test_constructor_preserves_new_class_defaults_and_applies_typed_base_config(processor_module, training):
    module = processor_module
    selected = processor({'image_horizon': 3, 'training_mode': False, 'action_sequence_keys': ['actions', 'state'],
                          'base_config.prompt_from_task': True, 'base_config.action_loss_mask': [1, 0]})
    value = instantiate_data_processor(selected, module, training[1]['code_dir'], dataset_path='/selected/data', norm_dir='/selected/norm')
    assert value.image_horizon == 3 and not value.training_mode and value.temperature == 0.1
    assert value.mapping == {'images': 'observation.camera'} and module.CALLS == ['mapping-default']
    assert value.repo_id == '/selected/data' and value.assets.assets_dir == '/selected/norm'
    assert value.base_config.prompt_from_task and value.base_config.action_loss_mask == [1, 0]
    assert validate_processor_instance(value, selected) == selected


@pytest.mark.parametrize('parameters', [{'image_horizon': True}, {'image_horizon': 2.5}, {'training_mode': 'false'},
                                       {'mode': 'missing'}, {'unknown': 1}, {'base_config.prompt_from_task': None},
                                       {'action_sequence_keys': [1]}])
def test_bad_parameter_types_never_reach_constructor(processor_module, training, parameters):
    with pytest.raises(ValueError, match='DATA_PROCESSOR_PARAMETERS_INVALID'):
        instantiate_data_processor(processor(parameters), processor_module, training[1]['code_dir'])
    assert processor_module.CALLS == []


def test_required_fields_and_complex_constructor_are_explicit(processor_module, training):
    with pytest.raises(ValueError, match='DATA_PROCESSOR_PARAMETERS_REQUIRED'):
        instantiate_data_processor(processor(name='RequiredFactory'), processor_module, training[1]['code_dir'])
    value = instantiate_data_processor(processor({'count': 3}, name='RequiredFactory'), processor_module, training[1]['code_dir'])
    assert value.count == 3
    with pytest.raises(ValueError, match='DATA_PROCESSOR_UNSUPPORTED'):
        instantiate_data_processor(processor(name='RequiredComplexFactory'), processor_module, training[1]['code_dir'])


def test_classes_outside_registered_checkout_are_never_available(processor_module, training, tmp_path):
    outside = tmp_path / 'other-checkout'
    outside.mkdir()
    assert all(not entry['available'] for entry in catalog_data_processors(processor_module, outside))
    with pytest.raises(ValueError, match='DATA_PROCESSOR_OUTSIDE_CODE_DIR'):
        instantiate_data_processor(processor(), processor_module, outside)


def test_runtime_independent_processors_replace_recipe_children_without_inheriting_settings(processor_module, training):
    base, spec, module, loaders = explicit(training, indices=(1, 1))
    module.DataConfigFactory = processor_module.DataConfigFactory
    module.DataConfig = processor_module.DataConfig
    module.IndependentFactory = processor_module.IndependentFactory
    # This factory is not present in the recipe; the old child's unrelated
    # settings must not bleed into the selected independent constructor.
    for child in base.data.datasets:
        child.image_horizon = 99
    for i, row in enumerate(spec['dataset_bindings']):
        row.pop('recipe_index')
        row['processor'] = processor({'image_horizon': i + 1, 'base_config.prompt_from_task': True})
    original = copy.deepcopy(base)
    config = resolve_config(base, spec, module, loaders)
    assert [child.image_horizon for child in config.data.datasets] == [1, 2]
    assert all(type(child) is processor_module.IndependentFactory for child in config.data.datasets)
    assert all(child.temperature == 0.1 for child in config.data.datasets)
    assert config.data.state_pad_dim == base.data.state_pad_dim and config.data.use_merged_norm_stats == base.data.use_merged_norm_stats
    assert base == original
    plan = inspect_training_config(config, spec, checks=[])
    assert plan['dataset_bindings'] == spec['dataset_bindings']
    assert all(data['processor'] == row['processor'] and 'recipe_index' not in data for data, row in zip(plan['datasets'], spec['dataset_bindings']))
    assert all(data['processor_identity']['source']['path'] == 'processor_config.py' for data in plan['datasets'])
    spec['dataset_bindings'][0]['processor']['parameters']['image_horizon'] = 3
    assert fingerprint(plan) != fingerprint(inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[]))
    config.data.datasets[0].image_horizon = 99
    with pytest.raises(ValueError, match='DATA_PROCESSOR_BINDING_MISMATCH'):
        inspect_training_config(config, spec, checks=[])


def test_single_recipe_can_select_an_independent_processor(processor_module, training):
    base, spec, module, loaders = explicit(training, indices=(0,))
    base.data = base.data.datasets[0]
    module.DataConfigFactory = processor_module.DataConfigFactory
    module.DataConfig = processor_module.DataConfig
    module.IndependentFactory = processor_module.IndependentFactory
    spec['dataset_bindings'][0].pop('recipe_index')
    spec['dataset_bindings'][0]['processor'] = processor()
    config = resolve_config(base, spec, module, loaders)
    assert type(config.data) is processor_module.IndependentFactory
    assert inspect_training_config(config, spec, checks=[])['datasets'][0]['processor'] == processor()


def test_conflicting_recap_stage_and_processor_parameter_has_explicit_error(training):
    base, spec, module, loaders = explicit(training, indices=(0,))
    spec.update(template_id='recap', training_stage='offline_rl', stage_config={'acp_dropout': 0.2})
    spec['dataset_bindings'][0].pop('recipe_index')
    spec['dataset_bindings'][0]['processor'] = processor({'acp_dropout': 0.5})
    with pytest.raises(ValueError, match='DATA_PROCESSOR_STAGE_CONFLICT'):
        resolve_config(base, spec, module, loaders)


def test_recap_stage_supplies_required_processor_constructor_parameter_without_changing_request(processor_module, training):
    from test_recap_runtime import Pi0Config, ReCapConfig
    old, spec, module, loaders = explicit(training, indices=(0,))
    base = ReCapConfig(**{field.name: copy.deepcopy(getattr(old, field.name)) for field in dataclasses.fields(old)})
    base.model = Pi0Config()
    module.DataConfigFactory = processor_module.DataConfigFactory
    module.DataConfig = processor_module.DataConfig
    module.RequiredACPFactory = processor_module.RequiredACPFactory
    spec.update(template_id='recap', training_stage='offline_rl', stage_config={'acp_dropout': 0.2})
    row = spec['dataset_bindings'][0]
    row.pop('recipe_index')
    row['processor'] = processor(name='RequiredACPFactory')
    original = copy.deepcopy(spec['dataset_bindings'])
    config = resolve_config(base, spec, module, loaders)
    assert config.data.datasets[0].acp_dropout == 0.2
    assert spec['dataset_bindings'] == original and row['processor']['parameters'] == {}
    assert validate_processor_instance(config.data.datasets[0], row['processor']) == row['processor']


def test_catalog_cli_uses_local_source_imports_and_cpu_environment(tmp_path):
    code = tmp_path / 'code'
    package = code / 'src/openpi/training'
    package.mkdir(parents=True)
    (package / 'config.py').write_text(SOURCE + '\nimport os\nassert os.environ["CUDA_VISIBLE_DEVICES"] == ""\nassert os.environ["JAX_PLATFORMS"] == "cpu"\nassert not CALLS\n')
    result = subprocess.run([sys.executable, '-B', str(Path(data_processors.__file__)), '--catalog-json',
                             json.dumps({'code_dir': str(code), 'allowed_roots': [str(tmp_path)]})], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    response = json.loads(result.stdout.split('VLA_DATA_PROCESSORS:')[-1])
    assert response['source']['path'] == 'src/openpi/training/config.py'
    assert any(row['name'] == 'IndependentFactory' and row['available'] for row in response['entries'])


def test_catalog_never_evaluates_annotation_expressions(processor_module, training):
    processor_module.IndependentFactory.__dataclass_fields__['image_horizon'].type = "__import__('os').system('false')"
    entries = catalog_data_processors(processor_module, training[1]['code_dir'])
    item = next(row for row in entries if row['name'] == 'IndependentFactory')
    assert next(row for row in item['parameters'] if row['path'] == 'image_horizon')['default'] == 2
    assert processor_module.CALLS == []
