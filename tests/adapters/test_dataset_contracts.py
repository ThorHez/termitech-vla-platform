"""Dataset recommendations use bounded source hints and real metadata evidence."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from vla_adapters.data_processors import catalog_data_processors
from vla_adapters.dataset_contracts import (contract_from_class_nodes, normalize_dataset_contract,
                                           static_dataset_contract, verify_registered_dataset_profile)
from vla_adapters import training_plan
from test_training_plan import training  # noqa: F401


SOURCE = '''from __future__ import annotations
import abc,dataclasses
CALLS=[]
@dataclasses.dataclass
class DataConfigFactory(abc.ABC):
    @abc.abstractmethod
    def create(self):pass
@dataclasses.dataclass
class Concrete(DataConfigFactory):
    mapping:dict[str,str]=dataclasses.field(default_factory=lambda:{
       'robot0_eef_pos':'observation.robot0_eef_pos',
       'robot0_eef_rot_axis_angle':'observation.robot0_eef_rot_axis_angle',
       'wrist_rgb':'observation.images.wrist','actions':'actions','prompt':'task'})
    def create(self):
        CALLS.append('create')
        repack=_transforms.Group(inputs=[_transforms.RepackTransform(self.mapping)])
        model_transforms=ModelTransformFactory()(None)
        return dataclasses.replace(None,repack_transforms=repack)
@dataclasses.dataclass
class Dual(Concrete):
    mapping:dict[str,str]=dataclasses.field(default_factory=lambda:{
       'robot0_eef_pos':'robot0_eef_pos','robot1_eef_pos':'robot1_eef_pos',
       'left_wrist_rgb':'left_wrist_rgb','right_wrist_rgb':'right_wrist_rgb',
       'actions':'actions','prompt':'task'})
@dataclasses.dataclass
class Joint(DataConfigFactory):
    def create(self):
        repack=_transforms.Group(inputs=[_transforms.RepackTransform({
            'robot0_joint_pos':'observation.joints','camera0_rgb':'observation.images.camera',
            'actions':'actions','prompt':'task'})])
        model_transforms=ModelTransformFactory()(None)
        return dataclasses.replace(None,repack_transforms=repack)
def dangerous_mapping():
    CALLS.append('default-factory')
    raise AssertionError('must not call')
@dataclasses.dataclass
class Dynamic(Concrete):
    mapping:dict[str,str]=dataclasses.field(default_factory=dangerous_mapping)
@dataclasses.dataclass
class Inference(Concrete):pass
Alias=Concrete
'''


@pytest.fixture
def source_module(tmp_path, monkeypatch):
    path=tmp_path/'factories.py';path.write_text(SOURCE)
    spec=importlib.util.spec_from_file_location('dataset_contract_factory_fixture',path)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module);spec.loader.exec_module(module)
    return module,tmp_path


def test_catalog_reads_literal_and_inherited_defaults_without_executing_them(source_module):
    module,root=source_module
    entries={entry['name']:entry for entry in catalog_data_processors(module,root)}
    single=entries['Concrete']['dataset_contract'];dual=entries['Dual']['dataset_contract']
    assert single['arm_mode']=='single' and single['action_representation']=='eef'
    assert single['camera_keys']==['observation.images.wrist']
    assert single['model_family']=='standard' and single['training_stage']=='sft'
    assert dual['arm_mode']=='bimanual' and dual['camera_keys']==['left_wrist_rgb','right_wrist_rgb']
    assert entries['Joint']['dataset_contract']['action_representation']=='joint'
    assert 'dataset_contract' not in entries['Dynamic']
    assert 'dataset_contract' not in entries['Inference']
    assert 'Alias' not in entries and module.CALLS==[]
    assert 'action_mode' not in single and 'rotation_representation' not in single


def test_actual_openpi_umi_single_bimanual_and_hybrid_mapping_sources():
    root=Path(__file__).resolve().parents[2]
    source=root/'third_party/openpi-umi/src/openpi/training/config.py'
    nodes={node.name:node for node in ast.parse(source.read_text()).body if isinstance(node,ast.ClassDef)}
    single=contract_from_class_nodes([nodes['LeRobotUmiDataConfigPadded_V4_ImageHorizon1']])
    assert single['camera_keys']==['observation.left_wrist_0_rgb_0']
    assert single['arm_mode']=='single' and single['model_family']=='standard'
    dual=contract_from_class_nodes([nodes['LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1']])
    assert dual['arm_mode']=='bimanual' and dual['camera_keys']==['left_wrist_0_rgb_0','right_wrist_0_rgb_0']
    hybrid=contract_from_class_nodes([nodes['LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid'],nodes['LeRobotUmiDataConfig_Hybrid']])
    assert hybrid['model_family']=='hybrid' and hybrid['arm_mode']=='bimanual'
    # Dynamic frame expansion cannot be certified by literal AST inspection.
    assert contract_from_class_nodes([nodes['LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint']]) is None
    for name in ('LeRobotUmiDataConfigPadded_V4_Inference','Shellgame_Compute_Norm_Stats'):
        assert contract_from_class_nodes([nodes[name]]) is None


@pytest.mark.parametrize('replacement',[
    "mapping={'robot0_eef_pos':choose()}",
    "mapping={**other}",
    "mapping={'robot0_eef_pos':'x','robot0_eef_pos':'y'}",
])
def test_dynamic_or_duplicate_dictionary_fields_are_not_hints(replacement):
    source='''class Factory:
    MAPPING
    def create(self):
        repack=Group(inputs=[RepackTransform(self.mapping)])
        return dataclasses.replace(None,repack_transforms=repack)
'''.replace('MAPPING',replacement)
    assert contract_from_class_nodes([ast.parse(source).body[0]]) is None


@pytest.mark.parametrize('mutation',[
    'repack=other', 'repack.mutate()',
    'if condition: repack=other',
])
def test_repack_mutations_do_not_grant_hints(mutation):
    source='''class Factory:
    def create(self):
        repack=Group(inputs=[RepackTransform({'robot0_eef_pos':'x','camera_rgb':'rgb','actions':'actions'})])
        MUTATION
        return dataclasses.replace(None,repack_transforms=repack)
'''.replace('MUTATION',mutation)
    assert contract_from_class_nodes([ast.parse(source).body[0]]) is None


def test_rebound_method_or_outside_source_is_not_advertised(source_module,tmp_path):
    module,root=source_module
    assert static_dataset_contract(module.Concrete,tmp_path/'outside') is None
    original=module.Concrete.create
    try:
        module.Concrete.create=module.Joint.create
        assert static_dataset_contract(module.Concrete,root) is None
    finally:module.Concrete.create=original


def profile():
    return {'schema_version':1,'format':'lerobot','robot_type':'ARX5','arm_mode':'single',
            'action_representation':'eef','cameras':[{'name':'腕部','key':'observation.images.wrist','kind':'rgb'}],
            'task_description':'将杯子放到托盘','action_dim':10,'fps':30,
            'feature_keys':['observation.images.wrist','actions']}


def info():
    return {'fps':30,'features':{'actions':{'dtype':'float32','shape':[16,10]},
            'observation.images.wrist':{'dtype':'image','shape':[224,224,3]}}}


def test_runtime_profile_checks_real_metadata_and_returns_unaliased_json_copy():
    original=profile();actual=verify_registered_dataset_profile(original,info())
    assert actual==original and actual is not original
    actual['cameras'][0]['name']='changed'
    assert original['cameras'][0]['name']=='腕部'


def test_multiline_task_description_remains_metadata_and_preserves_original_json():
    declared=profile();declared['task_description']='第一步：拿杯子\n第二步：\t放到托盘'
    assert verify_registered_dataset_profile(declared,info())==declared
    declared['task_description']='禁止\x00控制符'
    with pytest.raises(ValueError,match='task_description'):
        verify_registered_dataset_profile(declared,info())


@pytest.mark.parametrize('field,value', [('action_dim',9),('fps',0),('state_dim',16),('format','rlds'),
    ('schema_version',2),('action_dim',True),('feature_keys',['task']),('feature_keys',['missing']),
    ('feature_keys',['actions','actions']),('fps',float('inf')),('robot_type','bad\nrobot')])
def test_declared_profile_cannot_override_actual_metadata(field,value):
    declared=profile();declared[field]=value
    with pytest.raises(ValueError):verify_registered_dataset_profile(declared,info())


def test_valid_fps_difference_is_a_warning_without_modifying_registration_or_metadata():
    declared, document = profile(), info()
    declared['fps'] = 20;document['fps'] = 60
    before = copy.deepcopy((declared, document))
    warnings = [{'code': 'EXISTING_WARNING'}]
    result = verify_registered_dataset_profile(declared, document, warnings=warnings)
    assert result == declared and result is not declared
    assert (declared, document) == before
    assert warnings[0] == {'code': 'EXISTING_WARNING'}
    warning = warnings[1]
    assert warning['code'] == 'DATASET_FPS_MISMATCH'
    assert warning['supplied'] == 20 and warning['metadata'] == 60
    assert warning['effective'] == 20 and warning['source'] == 'registered'
    assert '本次训练进程内采用登记采样率 20 Hz' in warning['message']
    assert '不自动重采样、修改时间戳或写回元信息文件' in warning['message']
    # Older direct callers need no collector and still receive the profile only.
    assert verify_registered_dataset_profile(declared, document) == declared


@pytest.mark.parametrize('fps', [30, 30.0, 30 + 1e-7])
def test_equal_or_numerically_equivalent_fps_adds_no_warning(fps):
    declared = profile();declared['fps'] = fps
    warnings = []
    assert verify_registered_dataset_profile(declared, info(), warnings=warnings) == declared
    assert warnings == []


@pytest.mark.parametrize('fps', [0, -1, True, None, '30', 1001, float('nan'), float('inf'), 10**400])
def test_invalid_declared_fps_still_blocks_instead_of_becoming_a_warning(fps):
    declared = profile();declared['fps'] = fps
    warnings = []
    with pytest.raises(ValueError, match='DATASET_PROFILE_MISMATCH'):
        verify_registered_dataset_profile(declared, info(), warnings=warnings)
    assert warnings == []


@pytest.mark.parametrize('fps', [0, -1, True, None, '30', float('nan'), float('inf'), 10**400])
def test_registered_fps_still_requires_usable_actual_metadata(fps):
    document = info();document['fps'] = fps
    warnings = []
    with pytest.raises(ValueError, match='DATASET_PROFILE_MISMATCH'):
        verify_registered_dataset_profile(profile(), document, warnings=warnings)
    assert warnings == []


def test_missing_actual_fps_still_blocks_only_the_existing_registered_fps_path():
    document = info();document.pop('fps')
    with pytest.raises(ValueError, match='DATASET_PROFILE_MISMATCH'):
        verify_registered_dataset_profile(profile(), document, warnings=[])
    declared = profile();declared.pop('fps')
    assert verify_registered_dataset_profile(declared, document, warnings=[]) == declared


def test_fps_warning_does_not_weaken_real_action_tensor_dimension_checks():
    declared, document = profile(), info()
    declared['fps'] = 60
    document['features']['actions']['shape'] = [16, 9]
    with pytest.raises(ValueError, match='action_dim'):
        verify_registered_dataset_profile(declared, document, warnings=[])


def test_state_dim_requires_one_explicit_vector_and_depth_requires_real_semantics():
    declared=profile();declared['state_dim']=16
    with pytest.raises(ValueError,match='DATASET_PROFILE_UNVERIFIABLE'):
        verify_registered_dataset_profile(declared,info())
    document=info();document['features']['observation.state']={'shape':[16]}
    assert verify_registered_dataset_profile(declared,document)==declared
    document['features']['state']={'shape':[17]}
    with pytest.raises(ValueError,match='state_dim'):verify_registered_dataset_profile(declared,document)
    declared=profile();declared['cameras'][0]['kind']='depth'
    with pytest.raises(ValueError,match='深度'):verify_registered_dataset_profile(declared,info())
    document=info();document['features']['observation.images.wrist']['info']={'is_depth':True}
    assert verify_registered_dataset_profile(declared,document)==declared
    declared['cameras'][0]['kind']='rgb'
    with pytest.raises(ValueError,match='RGB'):verify_registered_dataset_profile(declared,document)


def test_derived_task_requires_explicit_verified_task_index_evidence():
    declared=profile();declared['feature_keys'].append('task')
    document=info();document['features']['task_index']={'dtype':'int64','shape':[1]}
    with pytest.raises(ValueError,match='task'):verify_registered_dataset_profile(declared,document)
    assert verify_registered_dataset_profile(declared,document,derived_features=['task'])==declared
    with pytest.raises(ValueError):verify_registered_dataset_profile(declared,document,derived_features=['made_up'])


def test_legacy_training_plan_omits_profile_and_registered_plan_freezes_verified_profile(training):
    base,spec,module,loaders=training
    from vla_adapters.openpi_entry import resolve_config
    resolved=resolve_config(base,spec,module,loaders)
    legacy=training_plan.inspect_training_config(resolved,spec,checks=[])
    assert all('registered_profile' not in item for item in legacy['datasets'])
    selected=Path(spec['inputs']['dataset_1']['path'])
    document=json.loads((selected/'meta/info.json').read_text())
    document.update(info());(selected/'meta/info.json').write_text(json.dumps(document))
    declared=profile();spec['inputs']['dataset_1']['dataset_profile']=declared
    plan=training_plan.inspect_training_config(resolved,spec,checks=[])
    assert plan['datasets'][1]['registered_profile']==declared
    assert all('registered_profile' not in plan['datasets'][index] for index in (0,2))
    declared['action_dim']=99
    with pytest.raises(ValueError,match='DATASET_PROFILE_MISMATCH'):
        training_plan.inspect_training_config(resolved,spec,checks=[])


@pytest.mark.parametrize('single', [False, True])
def test_fps_warning_survives_single_or_multi_plan_and_repeated_runner_validation(training, single):
    base, spec, module, loaders = training
    from vla_adapters.openpi_entry import resolve_config
    selected = spec['inputs']['dataset_1']
    path = Path(selected['path']);metadata_path = path/'meta/info.json'
    document = json.loads(metadata_path.read_text());document.update(info());document['fps'] = 60
    metadata_path.write_text(json.dumps(document))
    declared = profile();declared['fps'] = 20
    selected['dataset_profile'] = declared
    if single:
        base.data = base.data.datasets[0]
        spec['inputs'] = {'dataset': selected, 'norm': spec['inputs']['norm_1'], 'model': spec['inputs']['model']}
        spec.pop('dataset_bindings')
    config = resolve_config(base, spec, module, loaders)
    before_metadata, before_spec = metadata_path.read_bytes(), copy.deepcopy(spec)
    checks = []
    plan = training_plan.inspect_training_config(config, spec, checks=checks)
    data = plan['datasets'][0 if single else 1]
    warning = data['warnings'][0]
    assert warning['code'] == 'DATASET_FPS_MISMATCH'
    assert warning['supplied'] == 20 and warning['metadata'] == 60
    assert warning['effective'] == 20 and warning['source'] == 'registered'
    assert warning['dataset_path'] == str(path) and str(path) in warning['message']
    assert data['registered_profile']['fps'] == 20
    warning_checks = [check for check in checks if check.get('severity') == 'warning']
    assert warning_checks == [{'name': 'dataset_fps', 'ok': True, 'severity': 'warning',
                               'code': 'DATASET_FPS_MISMATCH', 'message': warning['message']}]
    assert all(check['ok'] for check in checks)
    assert {'data_contract', 'norm_contract', 'model_checkpoint', 'training_entrypoint', 'training_cache'} <= {
        check['name'] for check in checks}
    assert metadata_path.read_bytes() == before_metadata and spec == before_spec
    # Runner uses the same deterministic inspection; output paths do not change it.
    spec['output_dir'] += '-runner'
    repeated = training_plan.inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert training_plan.fingerprint(repeated) == training_plan.fingerprint(plan)
    document['fps'] = 30;metadata_path.write_text(json.dumps(document))
    changed = training_plan.inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert changed['datasets'][0 if single else 1]['warnings'][0]['metadata'] == 30
    assert training_plan.fingerprint(changed) != training_plan.fingerprint(plan)


def test_fps_warning_does_not_hide_subsequent_normalization_failure(training):
    base, spec, module, loaders = training
    from vla_adapters.openpi_entry import resolve_config
    path = Path(spec['inputs']['dataset_0']['path'])
    metadata_path = path/'meta/info.json'
    document = json.loads(metadata_path.read_text());document.update(info())
    metadata_path.write_text(json.dumps(document))
    declared = profile();declared['fps'] = 60
    spec['inputs']['dataset_0']['dataset_profile'] = declared
    config = resolve_config(base, spec, module, loaders)
    (path/'norm_stats.json').write_text('{"norm_stats":{"state":{},"actions":{}}}')
    checks = []
    with pytest.raises(ValueError, match='NORM_FORMAT'):
        training_plan.inspect_training_config(config, spec, checks=checks)
    assert any(check.get('code') == 'DATASET_FPS_MISMATCH' and check['ok'] is True for check in checks)


def test_validate_only_cli_exposes_fps_warning_and_keeps_later_checks(training, monkeypatch, capsys):
    from contextlib import nullcontext
    from vla_adapters import openpi_entry
    base, spec, module, loaders = training
    path = Path(spec['inputs']['dataset_0']['path'])
    metadata_path = path/'meta/info.json'
    document = json.loads(metadata_path.read_text());document.update(info())
    metadata_path.write_text(json.dumps(document))
    declared = profile();declared['fps'] = 60
    spec['inputs']['dataset_0']['dataset_profile'] = declared
    config = openpi_entry.resolve_config(base, spec, module, loaders)
    monkeypatch.setattr(openpi_entry, 'prepare_config', lambda _: (config, None))
    # This test isolates CLI warning propagation; runtime Metadata adaptation
    # and its constructor contract are covered by the FPS runtime tests.
    monkeypatch.setattr(openpi_entry, 'registered_fps_context', lambda _: nullcontext(None))
    monkeypatch.setattr(sys, 'argv', ['openpi_entry', '--spec-json', json.dumps(spec), '--validate-only'])
    openpi_entry.main()
    result = json.loads(capsys.readouterr().out.split('VLA_PREFLIGHT:')[-1])
    assert 'error' not in result
    assert any(check.get('severity') == 'warning' and check['name'] == 'dataset_fps'
               and check['ok'] is True for check in result['checks'])
    assert any(check['name'] == 'training_cache' and check['ok'] is True for check in result['checks'])
    assert not Path(spec['output_dir']).exists()


def test_lerobot_v2_tasks_file_is_required_for_declared_derived_task(training):
    base,spec,_,_=training;factory=base.data.datasets[0];path=Path(factory.repo_id)
    document=json.loads((path/'meta/info.json').read_text());document.update(info())
    document['features']['task_index']={'dtype':'int64','shape':[1]}
    (path/'meta/info.json').write_text(json.dumps(document))
    declared=profile();declared['feature_keys'].append('task')
    with pytest.raises(ValueError,match='真实 meta/tasks.jsonl'):
        training_plan.dataset_contract(factory,spec,declared)
    tasks=path/'meta/tasks.jsonl';tasks.write_text(json.dumps({'task_index':0,'task':'整理杯子'})+'\n')
    checked=training_plan.dataset_contract(factory,spec,declared)
    assert checked['derived_features']==['task'] and len(checked['task_metadata_sha256'])==64
    assert checked['registered_profile']==declared
    tasks.write_text('{"task_index":0,"task":""}\n')
    with pytest.raises(ValueError,match='DATASET_PROFILE_UNVERIFIABLE'):
        training_plan.dataset_contract(factory,spec,declared)


@pytest.mark.parametrize('mutate',[
    lambda c:c.update(camera_keys=['not-required']),
    lambda c:c.update(camera_keys=['rgb','rgb']),
    lambda c:c.update(required_features=['a\x00b']),
    lambda c:c.update(required_features=['a']*257),
    lambda c:c.update(evidence='runtime_guaranteed'),
    lambda c:c.update(model_family='unknown'),
    lambda c:c.update(extra='not-whitelisted'),
])
def test_controller_boundary_rejects_invalid_optional_hints(mutate):
    from vla_control.data_processors import _entry
    contract={'arm_mode':'single','action_representation':'eef','camera_keys':['rgb'],
              'required_features':['rgb','actions'],'training_stage':'sft','evidence':'static_repack'}
    entry={'name':'Factory','description':'','parameters':[],'unsupported_parameters':[],
           'available':True,'source':{'path':'config.py','sha256':'a'*64},'dataset_contract':contract}
    assert _entry(entry)['dataset_contract']==normalize_dataset_contract(contract)
    mutate(contract)
    with pytest.raises(ValueError):_entry(entry)


def test_old_agent_entries_remain_compatible_and_untraceable_contracts_are_rejected():
    from vla_control.data_processors import _entry
    entry={'name':'Factory','description':'','parameters':[],'unsupported_parameters':[],'available':True}
    assert 'dataset_contract' not in _entry(entry)
    entry['dataset_contract']={}
    with pytest.raises(ValueError):_entry(entry)
