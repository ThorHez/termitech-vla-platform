import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createServer } from 'vite';

const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
try {
  const gpu = await server.ssrLoadModule('/src/trainingGpuSelection.ts');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const retry = await server.ssrLoadModule('/src/runRetry.ts');
  const templates = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const devices = [0,1,2].map((index) => ({ uuid:`GPU-${index}`, name:'A100', processes:[], utilization_pct:0,
    memory_used_mb:0, memory_total_mb:81920 }));
  const node = { id:'node-a',name:'Machine A',hostname:'host',status:'ONLINE',gpus:devices };
  const draft = () => ({ step:1, advancedRecipe:false, editedParameters:[],editedStageParameters:[],entrypointEdited:false,
    request:{ name:'Run',node_id:node.id,profile_id:'native',template_id:'openpi-sft',gpu_count:2,
      requested_gpu_uuids:['GPU-2','GPU-0'],inputs:{model:'model'},parameters:{batch_size:8},
      auto_config:{model_type:'pi05',finetune_mode:'full',action_horizon:50,action_dim:32},
      dataset_bindings:[{dataset_asset_id:'data',norm_asset_id:'norm',weight:1,
        processor:{name:'ComposableDataConfig',parameters:{composition:{}}}}] } });

  await test('pinned UUIDs survive draft round trips including an unfinished empty selection', () => {
    const source=draft();assert.deepEqual(codec.parseTrainingDraft(codec.stringifyTrainingDraft(source)),source);
    source.request.requested_gpu_uuids=[];source.request.gpu_count=0;
    assert.deepEqual(codec.parseTrainingDraft(codec.stringifyTrainingDraft(source)),source);
    assert.match(gpu.requestedGpuError(source.request,node),/至少选择/);
  });
  await test('invalid duplicate or oversized UUID lists cannot enter a persisted draft', () => {
    for(const selected of [['GPU-0','GPU-0'],['GPU-0\n'],Array.from({length:257},(_,i)=>'GPU-'+i),null]) {
      const source=draft();source.request.requested_gpu_uuids=selected;
      assert.throws(()=>codec.encodeTrainingDraft(source));
    }
  });
  await test('manual selection validates count and node identity without silently substituting a GPU', () => {
    assert.equal(gpu.requestedGpuError(draft().request,node),undefined);
    assert.match(gpu.requestedGpuError({...draft().request,gpu_count:1},node),/数量/);
    assert.match(gpu.requestedGpuError(draft().request,{gpus:[{uuid:'other-machine-GPU-2'}]}),/不属于/);
    assert.equal(gpu.requestedGpuError({gpu_count:2},undefined),undefined);
  });
  await test('busy and unknown cards remain selectable while their telemetry is clearly distinguished', () => {
    assert.equal(gpu.gpuSelectionState(devices[0]),'available');
    const busy={...devices[0],processes:[{pid:123}]};
    assert.equal(gpu.gpuSelectionState(busy),'busy');
    assert.equal(gpu.gpuSelectionState({...devices[0],reserved_by:'other-run'}),'busy');
    assert.equal(gpu.gpuSelectionState({...devices[0],processes:null}),'unknown');
    assert.equal(gpu.gpuSelectionState({...devices[0],memory_used_mb:null}),'unknown');
    assert.equal(gpu.requestedGpuError({gpu_count:1,requested_gpu_uuids:[busy.uuid]},{gpus:[busy]}),undefined);
  });
  await test('display numbering follows inventory while the persisted identity stays the exact UUID', () => {
    assert.equal(gpu.requestedGpuSummary(['GPU-2','GPU-0'],node),'GPU 2、GPU 0');
    assert.equal(gpu.requestedGpuSummary(['GPU-2'],{gpus:[devices[2],devices[0]]}),'GPU 0');
    assert.match(gpu.requestedGpuSummary(['missing'],node),/missing/);
  });
  await test('retry restores only requested GPU pins and never infers them from an old attempt assignment', () => {
    const source={...draft().request,id:'run-a',state:'FAILED',created_at:'2026-10-30',
      attempts:[{id:'attempt-a',gpu_uuids:['GPU-1','GPU-0']}]};
    assert.deepEqual(retry.retryTrainingSnapshot(source).request.requested_gpu_uuids,['GPU-2','GPU-0']);
    delete source.requested_gpu_uuids;
    assert.equal(retry.retryTrainingSnapshot(source).request.requested_gpu_uuids,undefined);
    source.requested_gpu_uuids=['GPU-2'];
    assert.throws(()=>retry.retryTrainingSnapshot(source),/指定 GPU/);
  });
  await test('saved templates keep pinned UUIDs on the same machine and report removed devices', () => {
    const content=templates.prepareTrainingTemplate(draft());
    const record={id:'template',name:'Pinned',revision:1,created_at:'now',updated_at:'now',content};
    const resources={nodes:[node],profiles:[{id:'native',node_id:node.id,adapter:'openpi-sft',execution_mode:'native'}],
      assets:[{id:'model',node_id:node.id,kind:'MODEL',state:'READY'}]};
    const applied=templates.applyTrainingTemplate(record,resources);
    assert.deepEqual(applied.snapshot.request.requested_gpu_uuids,['GPU-2','GPU-0']);
    assert.deepEqual(applied.issues,[]);
    resources.nodes=[{...node,gpus:[devices[1]]}];
    const missing=templates.applyTrainingTemplate(record,resources);
    assert.ok(missing.issues.some((issue)=>issue.includes('指定 GPU')));
    assert.deepEqual(missing.snapshot.request.requested_gpu_uuids,['GPU-2','GPU-0']);
  });
} finally { await server.close(); }
