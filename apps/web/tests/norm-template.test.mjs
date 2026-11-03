import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom',
});
try {
  const { NormStatsFields } = await server.ssrLoadModule('/src/NormStatsFields.tsx');
  const { NormDatasetEditor } = await server.ssrLoadModule('/src/NormDatasetEditor.tsx');
  const norm = await server.ssrLoadModule('/src/normStats.ts');
  const { composedProcessingErrors } = await server.ssrLoadModule('/src/composedProcessing.ts');
  const { compositionBindingState } = await server.ssrLoadModule('/src/ComposedDatasetProcessing.tsx');
  const { DEFAULT_AUTO_CONFIG } = await server.ssrLoadModule('/src/trainingConfiguration.ts');
  const { retryTrainingSnapshot } = await server.ssrLoadModule('/src/runRetry.ts');
  const { encodeTrainingDraft, decodeTrainingDraft } = await server.ssrLoadModule('/src/trainingDraft.ts');
  const { prepareTrainingTemplate, applyTrainingTemplate } = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const request = { name: 'Statistics', node_id: 'node', profile_id: 'profile', template_id: 'openpi-norm',
    recipe: 'pi05_data', inputs: { dataset: 'data' }, parameters: { batch_size: 32, num_workers: 0 }, gpu_count: 0 };

  await test('statistics fields explain source processing and require no model or existing norm', () => {
    const html = renderToStaticMarkup(React.createElement(NormStatsFields, { advanced:true, recipe: request.recipe, onChange() {} }));
    assert.match(html, /统计使用的代码配方/);
    assert.match(html, /单数据集配方/);
    assert.match(html, /选择多数据集配方中的数据项/);
    assert.match(html, /独立配置处理器/);
    assert.doesNotMatch(html, /单数据集代码配方|仅支持单数据集|只支持单数据集/);
    assert.match(html, /任务使用 CPU/);
    assert.match(html, /state \/ actions/);
    assert.match(html, /value="pi05_data"/);
    assert.doesNotMatch(html, /初始模型|选择归一化参数/);
  });

  await test('finished norm tasks restore dataset-only inputs with zero GPUs and no stale approval', () => {
    const restored = retryTrainingSnapshot({ ...request, id: 'run', state: 'SUCCEEDED',
      code_expectation: { confirmed_config_sha256: 'old-confirmation' } });
    assert.equal(restored.request.template_id, 'openpi-norm');
    assert.equal(restored.request.gpu_count, 0);
    assert.deepEqual(restored.request.inputs, { dataset: 'data' });
    assert.equal(restored.request.recipe, request.recipe);
    assert.equal(restored.request.confirmed_config_sha256, undefined);
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(restored)), restored);
    assert.throws(() => retryTrainingSnapshot({ ...request, id: 'run', state: 'SUCCEEDED', inputs: {} }), /数据集/);
  });

  await test('reusable norm templates keep OpenPI profile and statistics parameters while clearing dataset selection', () => {
    const snapshot = retryTrainingSnapshot({ ...request, id: 'run', state: 'SUCCEEDED' });
    const content = prepareTrainingTemplate(snapshot);
    const applied = applyTrainingTemplate({ content }, {
      nodes: [{ id: 'node', name: 'Node', status: 'ONLINE', gpus: [] }],
      profiles: [{ id: 'profile', node_id: 'node', adapter: 'openpi-sft', code_dir: '/code', python: '/python' }],
      assets: [],
    });
    assert.equal(applied.snapshot.request.profile_id, 'profile');
    assert.equal(applied.snapshot.request.gpu_count, 0);
    assert.deepEqual(applied.snapshot.request.inputs, {});
    assert.deepEqual(applied.snapshot.request.parameters, request.parameters);
    assert.equal(applied.issues.length, 0);
  });

  const rows = [{ dataset_asset_id: 'data-a', recipe_index: 0 },
    { dataset_asset_id: 'data-b', recipe_index: 1 }];
  const dataAssets = rows.map((row, index) => ({ id: row.dataset_asset_id, kind: 'DATASET', node_id: 'node', name: `Recorded dataset ${index+1}`, path: `/data/${index}`, version: '1', state:'READY' }));

  await test('batch statistics editor has independent dataset processing and no training asset or mixing controls', () => {
    const html = renderToStaticMarkup(React.createElement(NormDatasetEditor, {
      rows, assets:dataAssets, onChange() {}, recipe:'my_recipe',
    }));
    assert.match(html, /批量计算模式/);
    assert.match(html, /各自产出一份 norm_stats.json/);
    assert.match(html, /待统计数据集 1/); assert.match(html, /待统计数据集 2/);
    assert.match(html, /数据处理方式 1/); assert.match(html, /数据处理方式 2/);
    assert.match(html, /配方数据项序号 2/);
    assert.doesNotMatch(html, /选择归一化参数|混采权重|初始模型|基础模型|norm_asset_id/);
    const summary = renderToStaticMarkup(React.createElement(NormDatasetEditor, { rows, assets:dataAssets, onChange(){}, readOnly:true }));
    assert.match(summary, /Recorded dataset 1/); assert.match(summary, /Recorded dataset 2/);
    assert.match(summary, /配方数据项序号：1/); assert.match(summary, /配方数据项序号：2/);
    assert.doesNotMatch(summary, /<select|混采权重|选择归一化参数/);
  });

  await test('norm request boundary discards all training fields and preserves independent row choices', () => {
    const source = { ...request, norm_datasets: rows, inputs:{model:'old-model',norm:'old-norm'},
      dataset_bindings:[{dataset_asset_id:'data-a',norm_asset_id:'old-norm',weight:2,recipe_index:0}],
      normalization_method:'quantile', requested_gpu_uuids:['GPU-1'], gpu_count:1 };
    const preview = norm.buildNormPreviewRequest(source);
    assert.deepEqual(preview.inputs,{}); assert.deepEqual(preview.norm_datasets,rows); assert.equal(preview.gpu_count,0);
    for(const key of ['dataset_bindings','normalization_method','auto_config','requested_gpu_uuids']) assert.equal(Object.hasOwn(preview,key),false);
    assert.equal(norm.normDatasetErrors(rows).length,0);
    assert.match(norm.normDatasetErrors([rows[0],rows[0]]).join(''),/重复/);
    assert.match(norm.normDatasetErrors([{dataset_asset_id:'data',recipe_index:-1}]).join(''),/序号/);
    const appended=norm.appendNormDataset([{dataset_asset_id:'data',processor:{name:'Processor',parameters:{scale:2}}}]);
    assert.equal(appended[1].dataset_asset_id,''); appended[1].processor.parameters.scale=3;
    assert.equal(appended[0].processor.parameters.scale,2);
    const editorBinding=norm.normProcessingBinding(rows[0]);
    assert.deepEqual(norm.normProcessingChoice(editorBinding),rows[0]);
  });

  await test('norm batch drafts, personal templates and retry preserve rows without restoring norm or weights', () => {
    const batch={...request,inputs:{dataset_0:'data-a',dataset_1:'data-b'},norm_datasets:rows};
    const restored=retryTrainingSnapshot({...batch,id:'batch',state:'SUCCEEDED'});
    assert.deepEqual(restored.request.norm_datasets,rows); assert.deepEqual(restored.request.inputs,{});
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(restored)),restored);
    const content=prepareTrainingTemplate(restored);
    const applied=applyTrainingTemplate({content},{nodes:[{id:'node',name:'Node',status:'ONLINE'}],profiles:[{id:'profile',node_id:'node',adapter:'openpi-sft'}],assets:[]});
    assert.deepEqual(applied.snapshot.request.norm_datasets,rows.map(row=>({...row,dataset_asset_id:''})));
    assert.throws(()=>retryTrainingSnapshot({...batch,norm_datasets:[{...rows[0],weight:1}],id:'bad',state:'SUCCEEDED'}),/输入字段/);
    assert.throws(()=>encodeTrainingDraft({...restored,request:{...restored.request,norm_datasets:[{...rows[0],norm_asset_id:'old'}]}}));
    const customRows = [rows[0], {dataset_asset_id:'data-b',processor:{name:'ArmProcessor',parameters:{scale:2,action_delta:true}}}];
    const custom = retryTrainingSnapshot({...batch,norm_datasets:customRows,id:'custom',state:'SUCCEEDED'});
    assert.deepEqual(norm.buildNormPreviewRequest(custom.request).norm_datasets,customRows);
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(custom)).request.norm_datasets,customRows);
  });

  await test('default statistics configuration uses model shape and composition without a required code recipe', () => {
    const fields = renderToStaticMarkup(React.createElement(NormStatsFields, { recipe:'', autoConfig:DEFAULT_AUTO_CONFIG, onChange(){} }));
    for (const text of ['无需填写代码配方','动作预测长度','模型动作维度']) assert.match(fields,new RegExp(text));
    assert.doesNotMatch(fields,/模型结构|value="pi05"|value="pi0"/);
    assert.doesNotMatch(fields,/统计使用的代码配方|微调方式|选择归一化参数|选择初始模型|GPU 数量|训练步数|name="recipe"/);
    const fresh=norm.newNormDataset(true,dataAssets[0]);
    assert.equal(fresh.processor.name,'ComposableDataConfig'); assert.equal(fresh.recipe_index,undefined);
    assert.ok(composedProcessingErrors(fresh.processor.parameters.composition).length,'unfinished fields must not be accepted');
    const state=compositionBindingState({binding:norm.normProcessingBinding(fresh),dataset:dataAssets[0],entries:[],slots:[],index:0,stage:'sft'});
    assert.equal(state.valid,false);
    const editor=renderToStaticMarkup(React.createElement(NormDatasetEditor,{rows:[fresh],assets:dataAssets,automatic:true,onChange(){}}));
    assert.match(editor,/自定义字段组合/); assert.match(editor,/数据处理配置错误/);
    assert.doesNotMatch(editor,/配方数据项序号|当前配方数据项（兼容旧任务）|选择归一化参数|混采权重|初始模型/);
    assert.match(norm.normDatasetErrors(rows,true).join(''),/需配置独立处理器/);
  });

  await test('automatic statistics requests and retry round trips contain no code recipe or training assets', () => {
    const processors=[{dataset_asset_id:'data-a',processor:{name:'ArmProcessor',parameters:{scale:2}}},
      {dataset_asset_id:'data-b',processor:{name:'ArmProcessor',parameters:{scale:3}}}];
    const autoSource={...request,auto_config:{...DEFAULT_AUTO_CONFIG,action_horizon:20},norm_datasets:processors,
      inputs:{model:'stale-model',norm:'stale-norm'},parameters:{batch_size:32,num_workers:0,steps:100,learning_rate:0.1},gpu_count:1};
    const preview=norm.buildNormPreviewRequest(autoSource);
    assert.deepEqual(preview.auto_config,{...DEFAULT_AUTO_CONFIG,action_horizon:20});
    assert.equal(Object.hasOwn(preview,'recipe'),false); assert.deepEqual(preview.inputs,{});
    assert.deepEqual(preview.parameters,{batch_size:32,num_workers:0}); assert.equal(preview.gpu_count,0);
    assert.deepEqual(preview.norm_datasets,processors);
    const restored=retryTrainingSnapshot({...preview,recipe:'auto_pi05_full_h20_d32_sft',id:'auto',state:'SUCCEEDED'});
    assert.equal(restored.advancedRecipe,false); assert.equal(restored.request.recipe,undefined);
    assert.deepEqual(restored.request.auto_config,preview.auto_config);
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(restored)),restored);
    const content=prepareTrainingTemplate(restored);
    const applied=applyTrainingTemplate({content},{nodes:[{id:'node',name:'Node',status:'ONLINE'}],profiles:[{id:'profile',node_id:'node',adapter:'openpi-sft'}],assets:[]});
    assert.deepEqual(applied.snapshot.request.auto_config,preview.auto_config);
    assert.deepEqual(applied.snapshot.request.norm_datasets,processors.map(row=>({...row,dataset_asset_id:''})));
    assert.equal(norm.normAdvancedRecipe(restored.request,true),false);
    assert.throws(()=>retryTrainingSnapshot({...preview,auto_config:{...preview.auto_config,finetune_mode:'lora'},id:'invalid',state:'SUCCEEDED'}),/配置无效/);
  });

  await test('legacy named-recipe drafts keep advanced mode without changing SFT draft behavior', () => {
    assert.equal(norm.normAdvancedRecipe(request,false),true);
    assert.equal(norm.normAdvancedRecipe({...request,recipe:undefined,auto_config:DEFAULT_AUTO_CONFIG},false),false);
    assert.equal(norm.normAdvancedRecipe({...request,template_id:'openpi-sft'},false),false);
    assert.equal(norm.normAdvancedRecipe({...request,template_id:'openpi-sft'},true),true);
  });

  await test('statistics horizon suggestions retain per-dataset facts without training-mixture constraints', () => {
    const sources=dataAssets.map((asset,index)=>({...asset,metadata:{dataset_profile:{field_annotations:[{key:'actions',role:'action',shape:[index?32:16,8]}]}}}));
    const inferred=norm.inferNormActionHorizon(rows,sources);
    assert.deepEqual(inferred.errors,[]); assert.equal(inferred.recommendedHorizon,undefined);
    assert.deepEqual(inferred.sources.map(source=>source.sourceHorizon),[16,32]);
    assert.deepEqual(inferred.sources.map(source=>source.bindingIndex),[0,1]);
    assert.equal(norm.inferNormActionHorizon([rows[0]],sources).recommendedHorizon,16);
  });

  await test('norm data step hides duplicate automatic model fields while the parameter step keeps the editable controls', () => {
    const config={...DEFAULT_AUTO_CONFIG,action_horizon:NaN,action_dim:0};
    const render=(showAutomaticFields)=>renderToStaticMarkup(React.createElement(NormStatsFields,{recipe:'',autoConfig:config,showAutomaticFields,onChange(){}}));
    const dataStep=render(false), parameterStep=render(true);
    assert.doesNotMatch(dataStep,/模型结构|动作预测长度|模型动作维度|type="number"/);
    for(const field of ['动作预测长度','模型动作维度']) assert.match(parameterStep,new RegExp(field));
    assert.doesNotMatch(parameterStep,/模型结构|value="pi05"|value="pi0"/);
    assert.match(parameterStep,/type="number"/);
    assert.equal(config.action_dim,0);assert.ok(Number.isNaN(config.action_horizon),'hiding controls must not reset stored values');
    for(const model_type of ['pi0','pi05']) {
      const saved={...DEFAULT_AUTO_CONFIG,model_type,action_horizon:16,action_dim:24};
      renderToStaticMarkup(React.createElement(NormStatsFields,{recipe:'',autoConfig:saved,onChange(){},onAutoConfigChange(){assert.fail('render must preserve the saved model type');}}));
      assert.deepEqual(saved,{...DEFAULT_AUTO_CONFIG,model_type,action_horizon:16,action_dim:24});
    }
  });

} finally {
  await server.close();
}
