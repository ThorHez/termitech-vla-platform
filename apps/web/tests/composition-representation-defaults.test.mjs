import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false }, optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
try {
  const helper = await server.ssrLoadModule('/src/compositionRepresentationDefaults.ts');
  const processing = await server.ssrLoadModule('/src/composedProcessing.ts');
  const { newNormDataset, buildNormPreviewRequest } = await server.ssrLoadModule('/src/normStats.ts');
  const { ComposedProcessingEditor } = await server.ssrLoadModule('/src/ComposedProcessingEditor.tsx');
  const { fillRunRepresentationDefaults } = await server.ssrLoadModule('/src/ComposedDatasetProcessing.tsx');
  const { encodeTrainingDraft, decodeTrainingDraft } = await server.ssrLoadModule('/src/trainingDraft.ts');
  const profile = () => ({ schema_version:1, format:'lerobot', robot_type:'ARX5', arm_mode:'bimanual',
    action_representation:'unknown', action_mode:'unknown', rotation_representation:'unknown',
    cameras:[{name:'wrist',key:'observation.images.wrist',kind:'rgb'}], task_description:'Recorded numeric vectors',
    state_dim:14, action_dim:14, feature_keys:['observation.state','action'], field_annotations:[
      {key:'observation.state',role:'state',representation:'composite',component:'other',arm:'shared',value_mode:'absolute',shape:[14],dtype:'float32'},
      {key:'action',role:'action',representation:'composite',component:'other',arm:'shared',value_mode:'absolute',shape:[14],dtype:'float32'},
    ] });
  const dataset = (seed=profile()) => ({id:'data',node_id:'node',kind:'DATASET',name:'Recorded data',path:'/data/recorded',version:'1',metadata:{dataset_profile:seed}});
  const fieldHtml = (html,label) => html.slice(html.indexOf(label),html.indexOf('</label>',html.indexOf(label)));

  await test('norm copies explicit composite labels and source width into one shared block without inventing EEF or Joint', () => {
    const seed=profile(), before=structuredClone(seed);
    const spec=newNormDataset(true,dataset(seed)).processor.parameters.composition;
    assert.equal(spec.state_representation,'composite');assert.equal(spec.action_representation,'composite');
    assert.deepEqual(spec.action_layout,[{arm:'shared',kind:'shared',width:14,normalize:true}]);
    assert.deepEqual(spec.state_fields,[{key:'observation.state',width:14,arm:'shared',kind:'other',normalize:true}]);
    assert.deepEqual(spec.cameras,[]);assert.deepEqual(processing.composedProcessingErrors(spec,'norm'),[]);
    assert.ok(processing.composedProcessingErrors(spec).some(error=>error.includes('视觉输入')));
    const train=processing.trainingCompositionDraft(seed);
    assert.equal(train.state_representation,'composite');assert.equal(train.action_representation,'composite');
    assert.deepEqual(train.action_layout,spec.action_layout);
    assert.equal(train.cameras.length,1,'SFT keeps registered visual inputs');
    assert.deepEqual(seed,before);
    seed.field_annotations[1].shape=undefined;
    assert.equal(processing.trainingCompositionDraft(seed,'norm').action_layout[0].width,0,'missing source width must never be guessed');
  });

  await test('SFT and norm display inherited composite choices while retaining their distinct visual and normalization controls', () => {
    const seed=profile(), value=processing.trainingCompositionDraft(seed,'norm');
    const norm=renderToStaticMarkup(React.createElement(ComposedProcessingEditor,{value,profile:seed,purpose:'norm',onChange(){}}));
    const sft=renderToStaticMarkup(React.createElement(ComposedProcessingEditor,{value,profile:seed,onChange(){}}));
    for(const label of ['状态表示','组合动作表示']) {
      assert.match(fieldHtml(norm,label),/value="composite" selected="">组合向量（多分量）/);
      assert.match(fieldHtml(sft,label),/value="composite" selected="">组合向量（多分量）/);
    }
    assert.match(norm,/observation.state 的登记标签/);assert.match(norm,/action 的登记标签/);
    assert.doesNotMatch(norm,/归一化状态字段|归一化动作分块|组合归一化方式|模型视觉槽位/);
    assert.match(sft,/归一化状态字段|归一化动作分块/);
    assert.match(sft,/视觉输入与模型槽位/);
  });

  await test('old blank choices are filled from selected labels without changing the existing shared layout or normalization settings', () => {
    const seed=profile(), spec=processing.trainingCompositionDraft(seed,'norm');
    spec.state_representation='';spec.action_representation='';spec.normalization='zscore';
    spec.action_layout[0].normalize=false;spec.state_fields[0].normalize=false;
    const before=structuredClone(spec);
    const filled=helper.fillMissingCompositionRepresentations(spec,seed,'norm');
    assert.deepEqual(filled,{...before,state_representation:'composite',action_representation:'composite'});
    assert.deepEqual(spec,before);
    const manual=helper.fillMissingCompositionRepresentations(spec,seed,'norm',new Set(['state_representation']));
    assert.equal(manual.state_representation,'');assert.equal(manual.action_representation,'composite');
    const selected={...filled,state_representation:'joint'};
    assert.equal(helper.fillMissingCompositionRepresentations(selected,seed,'norm'),selected,'an existing choice is never replaced');
    const request=buildNormPreviewRequest({name:'Statistics',node_id:'node',profile_id:'profile',template_id:'openpi-norm',inputs:{},
      norm_datasets:[{dataset_asset_id:'data',processor:{name:'ComposableDataConfig',parameters:{composition:filled}}}],
      auto_config:{model_type:'pi05',finetune_mode:'full',action_horizon:50,action_dim:32},parameters:{batch_size:32,num_workers:8},gpu_count:0});
    const marker=helper.representationManualMarker(['node','data',0],'state_representation');
    const snapshot={request,step:0,advancedRecipe:false,editedParameters:[marker],editedStageParameters:[],entrypointEdited:false};
    const restored=decodeTrainingDraft(encodeTrainingDraft(snapshot));
    assert.deepEqual(restored.request.norm_datasets[0].processor.parameters.composition,filled);
    assert.ok(restored.editedParameters.includes(marker));
    const long=helper.representationManualMarker(['n'.repeat(200),'d'.repeat(200),0],'state_representation');
    assert.ok(long.length<=256);
    assert.ok(decodeTrainingDraft(encodeTrainingDraft({...snapshot,editedParameters:[long]})).editedParameters.includes(long));
  });

  await test('state and action infer independently and an action declaration cannot supply a missing state representation', () => {
    const seed=profile();seed.field_annotations[0].representation='joint';seed.field_annotations[1].representation='eef';
    let spec=processing.trainingCompositionDraft(seed);
    assert.equal(spec.state_representation,'joint');assert.equal(spec.action_representation,'eef');
    seed.field_annotations[0].representation='unknown';seed.action_representation='eef';
    spec=processing.trainingCompositionDraft(seed);
    assert.equal(spec.state_representation,'');assert.equal(spec.action_representation,'eef');
    assert.match(helper.inferCompositionRepresentations(seed,spec).state_representation.reason,/observation.state.*未知/);
  });

  await test('inference considers only selected source fields and reports conflicting or uncovered state labels by key', () => {
    const seed=profile();seed.field_annotations[0].representation='joint';seed.field_annotations[1].representation='eef';
    seed.field_annotations.push({...seed.field_annotations[0],key:'other.state',representation:'eef'});
    const spec=processing.trainingCompositionDraft(seed);spec.state_fields=spec.state_fields.slice(0,1);
    let evidence=helper.inferCompositionRepresentations(seed,spec);
    assert.equal(evidence.state_representation.value,'joint');
    spec.state_fields.push({key:'other.state',width:14,arm:'shared',kind:'other',normalize:true});
    evidence=helper.inferCompositionRepresentations(seed,spec);
    assert.equal(evidence.state_representation.value,undefined);assert.match(evidence.state_representation.reason,/冲突.*observation.state.*other.state/);
    seed.field_annotations[2].representation='unknown';
    evidence=helper.inferCompositionRepresentations(seed,spec);
    assert.equal(evidence.state_representation.value,undefined);assert.match(evidence.state_representation.reason,/other.state.*未知/);
    seed.field_annotations.push({...seed.field_annotations[1],key:'future.action',representation:'joint'});
    assert.equal(helper.inferCompositionRepresentations(seed,spec).action_representation.value,'eef');
    assert.equal(helper.inferCompositionRepresentations(seed,{...spec,action_key:'not_selected'}).action_representation.value,undefined);
  });

  await test('matching legacy processing supplies only missing semantics and conflicts are not resolved by majority', () => {
    const seed=profile(), selected=processing.trainingCompositionDraft(seed,'norm');
    for(const field of seed.field_annotations) field.representation='unknown';
    seed.processing={...structuredClone(selected),state_representation:'joint',action_representation:'eef',cameras:[]};
    const blank={...selected,state_representation:'',action_representation:''};
    const filled=helper.fillMissingCompositionRepresentations(blank,seed);
    assert.equal(filled.state_representation,'joint');assert.equal(filled.action_representation,'eef');
    assert.deepEqual(filled.action_layout,blank.action_layout);
    seed.field_annotations[0].representation='eef';
    const conflict=helper.inferCompositionRepresentations(seed,blank);
    assert.equal(conflict.state_representation.value,undefined);assert.match(conflict.state_representation.reason,/冲突.*observation.state.*已保存组合/);
    const other={...blank,state_fields:[{...blank.state_fields[0],key:'unrelated.state'}],action_key:'unrelated.action'};
    const unmatched=helper.inferCompositionRepresentations(seed,other);
    assert.equal(unmatched.state_representation.value,undefined);assert.equal(unmatched.action_representation.value,undefined);
  });

  await test('norm composite action requires the whole shared block and keeps real source width checks', () => {
    const seed=profile(), spec=processing.trainingCompositionDraft(seed,'norm');
    const mixed={...spec,action_layout:[...spec.action_layout,{arm:'left',kind:'joint',width:1,normalize:true}]};
    assert.match(processing.composedProcessingErrors(mixed,'norm').join(),/一个共享动作分块/);
    const wrong={...spec,action_layout:[{...spec.action_layout[0],width:13}]};
    assert.match(processing.trainingCompositionProfileErrors(seed,wrong,'norm').join(),/动作分块总宽度/);
  });

  await test('draft preparation fills blank representations before a preview and preserves explicit manual markers', () => {
    const seed=profile(), composition=processing.trainingCompositionDraft(seed,'norm');
    composition.state_representation='';composition.action_representation='';
    const request={name:'Statistics',node_id:'node',profile_id:'profile',template_id:'openpi-norm',inputs:{},
      norm_datasets:[{dataset_asset_id:'data',processor:{name:'ComposableDataConfig',parameters:{composition}}}],
      auto_config:{model_type:'pi05',finetune_mode:'full',action_horizon:50,action_dim:32},parameters:{batch_size:32,num_workers:8},gpu_count:0};
    const before=structuredClone(request), prepared=fillRunRepresentationDefaults(request,[dataset(seed)],new Set());
    assert.equal(prepared.norm_datasets[0].processor.parameters.composition.state_representation,'composite');
    assert.equal(prepared.norm_datasets[0].processor.parameters.composition.action_representation,'composite');
    assert.deepEqual(request,before);
    assert.equal(fillRunRepresentationDefaults(prepared,[dataset(seed)],new Set()),prepared,'already prepared configuration retains its identity');
    const manual=new Set(['state_representation','action_representation'].map(field=>helper.representationManualMarker(['node','data',0],field)));
    assert.equal(fillRunRepresentationDefaults(request,[dataset(seed)],manual),request);
  });

  await test('new SFT drafts inherit explicit composite fields as whole raw vectors and still require valid visual inputs', () => {
    const seed=profile(), before=structuredClone(seed);
    seed.image_features=[{key:'observation.images.wrist',dtype:'image',shape:[3,224,224]}];
    const spec=processing.trainingCompositionDraft(seed);
    assert.equal(spec.state_representation,'composite');assert.equal(spec.action_representation,'composite');
    assert.deepEqual(spec.action_layout,[{arm:'shared',kind:'shared',width:14,normalize:true}]);
    assert.deepEqual(spec.state_fields,[{key:'observation.state',width:14,arm:'shared',kind:'other',normalize:true}]);
    assert.equal(spec.cameras[0].key,'observation.images.wrist');assert.equal(spec.cameras[0].layout,'chw');
    assert.deepEqual(processing.composedProcessingErrors(spec),[]);
    assert.deepEqual(processing.trainingCompositionProfileErrors(seed,spec),[]);
    assert.ok(processing.composedProcessingErrors({...spec,cameras:[]}).some(error=>error.includes('视觉输入')));
    const split={...spec,action_layout:[{arm:'left',kind:'joint',width:7,normalize:true},{arm:'right',kind:'joint',width:7,normalize:true}]};
    assert.match(processing.composedProcessingErrors(split).join(),/一个共享动作分块/);
    const wrong={...spec,action_layout:[{...spec.action_layout[0],width:13}]};
    assert.match(processing.trainingCompositionProfileErrors(seed,wrong).join(),/动作分块总宽度/);
    assert.deepEqual(seed.field_annotations,before.field_annotations,'initialization never relabels source facts');
    seed.field_annotations[1].shape=undefined;
    assert.equal(processing.trainingCompositionDraft(seed).action_layout[0].width,0,'missing real width is not inferred from action_dim');
  });

  await test('SFT restoration fills only blank representations and keeps manual selections, normalization and existing layout', () => {
    const seed=profile(), composition=processing.trainingCompositionDraft(seed);
    composition.state_representation='';composition.action_representation='';composition.normalization='zscore';
    composition.action_layout[0].normalize=false;
    const request={name:'SFT',node_id:'node',profile_id:'profile',template_id:'openpi-sft',inputs:{model:'model'},
      dataset_bindings:[{dataset_asset_id:'data',norm_asset_id:'norm',weight:1,
        processor:{name:'ComposableDataConfig',parameters:{composition}}}],
      auto_config:{model_type:'pi05',finetune_mode:'full',action_horizon:50,action_dim:32},parameters:{batch_size:32},gpu_count:1};
    const before=structuredClone(request), prepared=fillRunRepresentationDefaults(request,[dataset(seed)],new Set());
    const restored=prepared.dataset_bindings[0].processor.parameters.composition;
    assert.equal(restored.state_representation,'composite');assert.equal(restored.action_representation,'composite');
    assert.deepEqual(restored.action_layout,composition.action_layout);
    assert.deepEqual(restored.cameras,composition.cameras);assert.equal(restored.normalization,'zscore');
    assert.deepEqual(request,before);
    const snapshot={request:prepared,step:0,advancedRecipe:false,editedParameters:[],editedStageParameters:[],entrypointEdited:false};
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(snapshot)),snapshot);
    const manual=new Set(['state_representation','action_representation'].map(field=>helper.representationManualMarker(['node','data',0],field)));
    assert.equal(fillRunRepresentationDefaults(request,[dataset(seed)],manual),request);
    const explicit={...restored,state_representation:'joint',action_representation:'eef',action_layout:[{arm:'right',kind:'position',width:14,normalize:false}]};
    assert.equal(helper.fillMissingCompositionRepresentations(explicit,seed),explicit,'existing user choices and layout remain intact');
  });

  await test('SFT never invents composite, EEF or Joint from a robot name, total width, mixed labels or conflicting saved semantics', () => {
    const seed=profile();
    for(const field of seed.field_annotations) field.representation='unknown';
    let spec=processing.trainingCompositionDraft(seed);
    assert.equal(spec.state_representation,'');assert.equal(spec.action_representation,'');assert.equal(spec.action_layout[0].width,0);
    seed.field_annotations[0].representation='eef';
    seed.field_annotations.push({...seed.field_annotations[0],key:'other.state',representation:'joint'});
    spec=processing.trainingCompositionDraft(seed);
    assert.equal(spec.state_representation,'');
    assert.match(helper.inferCompositionRepresentations(seed,spec).state_representation.reason,/冲突/);
    const known=profile(), knownSpec=processing.trainingCompositionDraft(known);
    known.processing={...structuredClone(knownSpec),state_representation:'joint',action_representation:'eef'};
    const evidence=helper.inferCompositionRepresentations(known,knownSpec);
    assert.equal(evidence.state_representation.value,undefined);assert.equal(evidence.action_representation.value,undefined);
    assert.match(evidence.state_representation.reason,/冲突/);assert.match(evidence.action_representation.reason,/冲突/);
    assert.equal(helper.fillMissingCompositionRepresentations({...knownSpec,state_representation:'',action_representation:''},known).state_representation,'');
  });
} finally { await server.close(); }
