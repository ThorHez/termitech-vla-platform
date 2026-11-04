import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const policy = await server.ssrLoadModule('/src/trainingGpuPolicy.ts');
  const selection = await server.ssrLoadModule('/src/trainingGpuSelection.ts');
  const codec = await server.ssrLoadModule('/src/trainingDraft.ts');
  const retry = await server.ssrLoadModule('/src/runRetry.ts');
  const templates = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const norm = await server.ssrLoadModule('/src/normStats.ts');
  const { TrainingGpuSelector } = await server.ssrLoadModule('/src/TrainingGpuSelector.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const shared = { mode: 'shared', memory_mb: 50 * 1024 };
  const device = {
    uuid: 'GPU-external-compute',
    name: 'NVIDIA A100 80GB',
    memory_total_mb: 80 * 1024,
    memory_used_mb: 20 * 1024,
    memory_free_mb: 60 * 1024,
    utilization_pct: 75,
    processes: [{ pid: 1234, kind: 'compute', ownership: 'external', memory_used_mb: 20 * 1024 }],
  };
  const node = {
    id: 'node-a',
    name: 'Machine A',
    hostname: 'host-a',
    status: 'ONLINE',
    gpus: [device],
    metrics: { capabilities: [policy.TRAINING_SHARED_GPU_CAPABILITY] },
  };
  const request = () => ({
    name: 'Shared training',
    node_id: node.id,
    profile_id: 'native',
    template_id: 'openpi-sft',
    gpu_count: 1,
    requested_gpu_uuids: [device.uuid],
    training_gpu_policy: { ...shared },
    inputs: { model: 'model' },
    parameters: { batch_size: 8, num_workers: 2 },
    auto_config: { model_type: 'pi05', finetune_mode: 'full', action_horizon: 50, action_dim: 32 },
    dataset_bindings: [
      {
        dataset_asset_id: 'dataset',
        norm_asset_id: 'norm',
        weight: 1,
        processor: { name: 'ComposableDataConfig', parameters: { composition: {} } },
      },
    ],
  });
  const snapshot = () => ({
    step: 1,
    advancedRecipe: false,
    editedParameters: [],
    editedStageParameters: [],
    entrypointEdited: false,
    request: request(),
  });
  const renderSelector = (value = request(), machine = node, userId = 'alice') =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { user: { id: userId } } },
        React.createElement(TrainingGpuSelector, {
          request: value,
          node: machine,
          minimum: 1,
          onChange() {},
        }),
      ),
    );

  await test('explicit sharing admits a known external compute process only within the remaining budget', () => {
    assert.equal(policy.sharedTrainingGpuError(device, shared), undefined);
    assert.equal(selection.gpuSelectionState(device, 'alice', shared), 'shareable');
    assert.equal(selection.requestedGpuError(request(), node, 1, 'alice'), undefined);
    assert.equal(policy.defaultTrainingMemoryBudget(node), 50 * 1024);
    assert.equal(policy.trainingGpuPolicySummary(shared), '共享显存 · 每卡预算 50.00 GiB');
  });

  await test('missing, invalid or contradictory GPU telemetry never permits shared training', () => {
    for (const changes of [
      { memory_total_mb: null },
      { memory_total_mb: 0 },
      { memory_used_mb: null },
      { memory_used_mb: -1 },
      { memory_free_mb: undefined },
      { memory_free_mb: NaN },
      { memory_free_mb: Infinity },
      { memory_free_mb: 61 * 1024 },
      { utilization_pct: null },
      { utilization_pct: -1 },
      { utilization_pct: 101 },
      { processes: null },
    ]) {
      const unknown = { ...device, ...changes };
      assert.match(policy.sharedTrainingGpuError(unknown, shared), /遥测不完整/);
      assert.equal(selection.gpuSelectionState(unknown, 'alice', shared), 'unknown');
      assert.match(
        selection.requestedGpuError(request(), { ...node, gpus: [unknown] }, 1, 'alice'),
        /遥测不完整/,
      );
    }
  });

  await test('unclassified processes and platform task occupancy remain blocked in shared mode', () => {
    for (const process of [
      { pid: 1234, ownership: 'external' },
      { pid: 1234, ownership: 'external', kind: 'unknown' },
      { pid: 0, ownership: 'external', kind: 'compute' },
      { pid: 1.5, ownership: 'external', kind: 'graphics' },
    ]) {
      const unknown = { ...device, processes: [process] };
      assert.match(policy.sharedTrainingGpuError(unknown, shared), /进程类型未知/);
      assert.equal(selection.gpuSelectionState(unknown, 'alice', shared), 'unknown');
    }
    for (const changes of [
      { reserved_by: 'running-platform-attempt' },
      { processes: [{ pid: 1234, kind: 'compute', ownership: 'platform' }] },
    ]) {
      const busy = { ...device, ...changes };
      assert.match(policy.sharedTrainingGpuError(busy, shared), /平台任务使用/);
      assert.equal(selection.gpuSelectionState(busy, 'alice', shared), 'busy');
      assert.match(
        selection.requestedGpuError(request(), { ...node, gpus: [busy] }, 1, 'alice'),
        /平台任务使用/,
      );
    }
  });

  await test('shared memory does not bypass another user reservation', () => {
    const reserved = {
      ...device,
      user_reservation: {
        user_id: 'alice',
        username: 'alice',
        display_name: 'Alice',
        created_at: 'now',
      },
    };
    const machine = { ...node, gpus: [reserved] };
    assert.equal(selection.gpuSelectionState(reserved, 'alice', shared), 'shareable');
    assert.equal(selection.requestedGpuError(request(), machine, 1, 'alice'), undefined);
    assert.equal(selection.gpuSelectionState(reserved, 'bob', shared), 'reserved');
    assert.match(selection.requestedGpuError(request(), machine, 1, 'bob'), /Alice.*预留/);
    const html = renderSelector(
      { ...request(), gpu_count: 0, requested_gpu_uuids: [] },
      machine,
      'bob',
    );
    assert.match(html, /aria-label="选择 GPU 0"[^>]*disabled=""/);
    assert.match(html, /Alice 已预留/);
  });

  await test('sharing requires the full memory budget plus reserve and rejects invalid budgets', () => {
    const free = shared.memory_mb + policy.TRAINING_GPU_RESERVE_MB;
    const fitting = {
      ...device,
      memory_free_mb: free,
      memory_used_mb: device.memory_total_mb - free,
    };
    assert.equal(policy.sharedTrainingGpuError(fitting, shared), undefined);
    const tooSmall = {
      ...fitting,
      memory_free_mb: free - 1,
      memory_used_mb: fitting.memory_used_mb + 1,
    };
    assert.match(policy.sharedTrainingGpuError(tooSmall, shared), /不足以容纳.*1 GiB 余量/);
    assert.equal(selection.gpuSelectionState(tooSmall, 'alice', shared), 'insufficient');
    assert.match(
      selection.requestedGpuError(request(), { ...node, gpus: [tooSmall] }, 1, 'alice'),
      /不足以容纳/,
    );
    for (const memory_mb of [NaN, Infinity, -1, 0, 255, 51200.5, '51200', 1024 * 1024 + 1]) {
      const invalid = { mode: 'shared', memory_mb };
      assert.equal(policy.validTrainingGpuPolicy(invalid), false);
      assert.match(policy.sharedTrainingGpuError(device, invalid), /有效的每卡显存预算/);
      assert.match(
        selection.requestedGpuError({ ...request(), training_gpu_policy: invalid }, node),
        /有效的每卡显存预算/,
      );
    }
  });

  await test('older Agents give a capability error and sharing always requires explicitly selected cards', () => {
    assert.equal(policy.supportsSharedTraining(node), true);
    for (const metrics of [{}, { capabilities: [] }, { capabilities: 'TRAINING_SHARED_GPU_V1' }]) {
      const oldNode = { ...node, metrics };
      assert.equal(policy.supportsSharedTraining(oldNode), false);
      assert.match(selection.requestedGpuError(request(), oldNode), /Agent.*不支持共享显存训练/);
    }
    for (const requested_gpu_uuids of [undefined, []])
      assert.match(
        selection.requestedGpuError({ ...request(), requested_gpu_uuids }, node),
        /需要明确勾选/,
      );
    assert.match(selection.requestedGpuError({ ...request(), gpu_count: 2 }, node), /数量/);
    assert.match(
      selection.requestedGpuError({ ...request(), requested_gpu_uuids: ['GPU-other-node'] }, node),
      /不属于当前机器/,
    );
  });

  await test('exclusive mode keeps busy cards queued and never relabels them as shareable', () => {
    const exclusive = request();
    delete exclusive.training_gpu_policy;
    assert.equal(selection.gpuSelectionState(device, 'alice'), 'busy');
    assert.equal(
      selection.requestedGpuError(exclusive, { ...node, metrics: {} }, 1, 'alice'),
      undefined,
    );
    const html = renderSelector(exclusive);
    assert.match(html, /占用中，将等待释放/);
    assert.doesNotMatch(html, /可共享使用|aria-label="每卡显存预算（GiB）"/);
    assert.equal(policy.trainingGpuPolicySummary(), '独占使用');
  });

  await test('GPU controls show mode, memory budget, remaining memory and capability failures', () => {
    const html = renderSelector();
    assert.match(html, /aria-label="GPU 使用模式"/);
    assert.match(html, /<option value="shared" selected="">共享剩余显存<\/option>/);
    assert.match(html, /aria-label="每卡显存预算（GiB）"[^>]*value="50"/);
    assert.match(html, /当前剩余 60\.0 GiB/);
    assert.match(html, /可共享使用/);
    assert.match(html, /<option value="automatic" disabled="">自动分配<\/option>/);
    assert.doesNotMatch(html, /role="alert"/);
    const legacy = renderSelector(request(), { ...node, metrics: {} });
    assert.match(legacy, /<option value="shared" disabled="" selected="">/);
    assert.match(legacy, /当前 Agent 不支持共享显存训练，更新后可使用。/);
    assert.match(legacy, /role="alert"[^>]*>当前 Agent 尚不支持共享显存训练/);
  });

  await test('drafts preserve shared policy and pinned identities as independent data', () => {
    const source = snapshot();
    const restored = codec.parseTrainingDraft(codec.stringifyTrainingDraft(source));
    assert.deepEqual(restored, source);
    assert.notEqual(restored.request.training_gpu_policy, source.request.training_gpu_policy);
    restored.request.training_gpu_policy.memory_mb = 32 * 1024;
    assert.equal(source.request.training_gpu_policy.memory_mb, 50 * 1024);
    for (const training_gpu_policy of [
      { mode: 'exclusive', memory_mb: 51200 },
      { mode: 'shared', memory_mb: '51200' },
      { mode: 'shared', memory_mb: 51200, bypass: true },
    ]) {
      const invalid = snapshot();
      invalid.request.training_gpu_policy = training_gpu_policy;
      assert.throws(() => codec.encodeTrainingDraft(invalid));
    }
  });

  await test('an unfinished NaN memory budget restores as blank but cannot pass GPU validation', () => {
    const unfinished = snapshot();
    unfinished.request.training_gpu_policy.memory_mb = NaN;
    const restored = codec.parseTrainingDraft(codec.stringifyTrainingDraft(unfinished));
    assert.ok(restored);
    assert.ok(Number.isNaN(restored.request.training_gpu_policy.memory_mb));
    assert.deepEqual(restored.request.requested_gpu_uuids, [device.uuid]);
    assert.match(selection.requestedGpuError(restored.request, node), /有效的每卡显存预算/);
    const html = renderSelector(restored.request);
    assert.match(html, /aria-label="每卡显存预算（GiB）"[^>]*value=""/);
    assert.match(html, /role="alert"[^>]*>请填写有效的每卡显存预算/);
    assert.doesNotMatch(html, /value="NaN"/);
  });

  await test('saved templates preserve sharing settings while requiring replacement dataset assets', () => {
    const source = snapshot();
    const content = templates.prepareTrainingTemplate(source);
    const record = {
      id: 'shared-template',
      name: 'Shared training',
      revision: 1,
      created_at: 'now',
      updated_at: 'now',
      content,
    };
    const resources = {
      nodes: [node],
      profiles: [
        { id: 'native', node_id: node.id, adapter: 'openpi-sft', execution_mode: 'native' },
      ],
      assets: [{ id: 'model', node_id: node.id, kind: 'MODEL', state: 'READY' }],
    };
    const applied = templates.applyTrainingTemplate(record, resources);
    assert.deepEqual(applied.snapshot.request.training_gpu_policy, shared);
    assert.deepEqual(applied.snapshot.request.requested_gpu_uuids, [device.uuid]);
    assert.deepEqual(applied.issues, []);
    assert.equal(applied.snapshot.request.dataset_bindings[0].dataset_asset_id, '');
    assert.equal(applied.snapshot.request.dataset_bindings[0].norm_asset_id, '');
    applied.snapshot.request.training_gpu_policy.memory_mb = 32 * 1024;
    assert.equal(source.request.training_gpu_policy.memory_mb, 50 * 1024);
    assert.equal(
      templates.applyTrainingTemplate(record, resources).snapshot.request.training_gpu_policy
        .memory_mb,
      50 * 1024,
    );
    const legacy = templates.applyTrainingTemplate(record, {
      ...resources,
      nodes: [{ ...node, metrics: {} }],
    });
    assert.ok(legacy.issues.some((issue) => /Agent.*不支持共享显存训练/.test(issue)));
    assert.deepEqual(legacy.snapshot.request.training_gpu_policy, shared);
  });

  await test('retry preserves explicit sharing rather than inferring it from previous GPU assignments', () => {
    const source = { ...request(), id: 'run-a', state: 'FAILED', created_at: 'now' };
    const restored = retry.retryTrainingSnapshot(source);
    assert.deepEqual(restored.request.training_gpu_policy, shared);
    assert.deepEqual(restored.request.requested_gpu_uuids, [device.uuid]);
    assert.notEqual(restored.request.training_gpu_policy, source.training_gpu_policy);
    const fromExpectation = { ...source, code_expectation: { training_gpu_policy: { ...shared } } };
    delete fromExpectation.training_gpu_policy;
    assert.deepEqual(
      retry.retryTrainingSnapshot(fromExpectation).request.training_gpu_policy,
      shared,
    );
    const missingPins = { ...source, attempts: [{ gpu_uuids: [device.uuid] }] };
    delete missingPins.requested_gpu_uuids;
    assert.throws(() => retry.retryTrainingSnapshot(missingPins), /共享显存配置无效/);
    assert.throws(
      () =>
        retry.retryTrainingSnapshot({
          ...source,
          training_gpu_policy: { mode: 'shared', memory_mb: NaN },
        }),
      /共享显存配置无效/,
    );
  });

  await test('CPU normalization requests discard training GPU pins and memory policy', () => {
    const { dataset_bindings, ...source } = request();
    source.template_id = 'openpi-norm';
    source.inputs = {};
    source.norm_datasets = [
      { dataset_asset_id: 'dataset', processor: dataset_bindings[0].processor },
    ];
    const normalized = norm.buildNormPreviewRequest(source);
    assert.equal(normalized.template_id, 'openpi-norm');
    assert.equal(normalized.gpu_count, 0);
    assert.equal(Object.hasOwn(normalized, 'training_gpu_policy'), false);
    assert.equal(Object.hasOwn(normalized, 'requested_gpu_uuids'), false);
    assert.equal(normalized.norm_datasets[0].dataset_asset_id, 'dataset');
    assert.deepEqual(source.training_gpu_policy, shared);
    assert.deepEqual(source.requested_gpu_uuids, [device.uuid]);
  });
} finally {
  await server.close();
}
