import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
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
  const { RecapWorkflow } = await server.ssrLoadModule('/src/RecapWorkflow.tsx');
  const { stageAdvantageCheckpoints, stageAdvantageHandoffSnapshot } = await server.ssrLoadModule(
    '/src/stageAdvantageHandoff.ts',
  );
  const { retryTrainingSnapshot } = await server.ssrLoadModule('/src/runRetry.ts');
  const { recapModelError, recapRecipeConfiguration, stageAnnotationCheckpointLabel } =
    await server.ssrLoadModule('/src/recap.ts');
  const { encodeTrainingDraft, decodeTrainingDraft } =
    await server.ssrLoadModule('/src/trainingDraft.ts');
  const { newRecapPhase, switchRecapPhase, replaceRecapPhase } =
    await server.ssrLoadModule('/src/recapPhaseDraft.ts');
  const { prepareTrainingTemplate } = await server.ssrLoadModule('/src/trainingTemplate.ts');
  const { recapValueAnnotationSnapshot } = await server.ssrLoadModule(
    '/src/recapValueAnnotationHandoff.ts',
  );
  const render = (selected, split, valueAnnotation = false) =>
    renderToStaticMarkup(
      React.createElement(RecapWorkflow, {
        selected,
        disabled: false,
        stages: [
          'value',
          'offline_rl',
          'stage_progress',
          ...(split ? ['stage_annotation'] : []),
          ...(valueAnnotation ? ['value_annotation'] : []),
        ].map((id) => ({
          id,
          backend: id === 'offline_rl' ? 'jax' : 'pytorch',
          default_entrypoint:
            id === 'offline_rl' ? 'scripts/train.py' : 'scripts/acp/train_recap_pytorch.py',
        })),
        onSelect() {},
      }),
    );

  await test('Stage Advantage shows independent jobs on a split-capable controller', () => {
    const html = render('stage_annotation', true);
    assert.match(html, /<strong>阶段进度训练<\/strong>/);
    assert.match(html, /aria-pressed="true"[^>]*>[\s\S]*?<strong>优势标注<\/strong>/);
    assert.match(html, /训练阶段 3 · JAX/);
    assert.doesNotMatch(html, /阶段进度训练与优势标注/);
    assert.match(render('stage_progress', false), /阶段进度训练与优势标注/);
    const standard = render('value', true);
    assert.match(standard, /优势值标注命令/);
    assert.match(standard, /训练阶段 2 · JAX/);
    assert.match(standard, /JAX \/ Orbax 初始策略权重/);
  });

  await test('a managed Value annotation has an independent workflow card while legacy copy commands remain available', () => {
    const html = render('value_annotation', true, true);
    assert.match(html, /aria-pressed="true"[^>]*>[\s\S]*?<strong>优势值标注<\/strong>/);
    assert.match(html, /训练阶段 3 · JAX/);
    assert.doesNotMatch(html, /优势值标注命令/);
  });

  await test('estimator training reads recipe validation controls while annotation uses its own parameters', () => {
    const values = { val_ratio: 0.2, eval_interval: 200, eval_batches: 5 };
    const recipe = {
      fields: Object.entries({ ...values, 'model.n_step': 100 }).map(([path, value]) => ({
        path,
        value,
        kind: 'literal',
      })),
    };
    for (const stage of ['value', 'stage_progress']) {
      assert.deepEqual(recapRecipeConfiguration(recipe, stage).values, values);
    }
    for (const stage of ['value_annotation', 'stage_annotation']) {
      assert.deepEqual(recapRecipeConfiguration(recipe, stage).values, {});
    }
  });

  await test('phase drafts preserve parameters, weights and recipe options independently across both ACP modes', () => {
    const value = newRecapPhase('value', 'node-value');
    value.request.inputs.model = 'value-start';
    value.request.parameters.steps = 20;
    let active = switchRecapPhase(value, 'value_annotation', []);
    assert.equal(active.request.node_id, '');
    assert.deepEqual(active.request.inputs, {});
    assert.deepEqual(active.request.parameters, {});
    active.request.stage_config = { n_step: 22, positive_ratio: NaN };
    active = switchRecapPhase(active, 'recap_acp', []);
    active.request.inputs.model = 'acp-recap';
    active = switchRecapPhase(active, 'stage_advantage_acp', []);
    active.request.inputs.model = 'acp-progress';
    assert.equal(active.request.training_stage, 'offline_rl');
    active = decodeTrainingDraft(encodeTrainingDraft(active));
    assert.equal(active.recapPhases.recap_acp.request.inputs.model, 'acp-recap');
    assert.equal(active.request.inputs.model, 'acp-progress');
    assert.equal(active.recapPhases.value.request.parameters.steps, 20);
    assert.ok(
      Number.isNaN(active.recapPhases.value_annotation.request.stage_config.positive_ratio),
    );
    active = switchRecapPhase(active, 'value', []);
    assert.equal(active.request.inputs.model, 'value-start');
    active = replaceRecapPhase(active, newRecapPhase('value'), []);
    assert.deepEqual(active.request.inputs, {});
    assert.equal(active.recapPhases.recap_acp.request.inputs.model, 'acp-recap');
    const template = prepareTrainingTemplate(active);
    assert.equal(template.recap_phase, 'value');
    assert.equal(template.recap_phases, undefined);
    assert.equal(
      decodeTrainingDraft({ ...encodeTrainingDraft(active), recap_phase: 'recap_acp' }),
      undefined,
    );
  });

  const fixture = () => {
    const asset = (id, kind) => ({ id, kind, node_id: 'node', state: 'READY' });
    const dataset = asset('dataset', 'DATASET');
    const norm = asset('norm', 'NORM');
    const checkpoint = {
      ...asset('checkpoint', 'MODEL'),
      origin_attempt_id: 'attempt',
      metadata: { format: 'openpi-pytorch', model_role: 'stage_progress_estimator' },
    };
    const run = {
      id: 'run',
      name: '原阶段进度',
      state: 'SUCCEEDED',
      node_id: 'node',
      profile_id: 'profile',
      template_id: 'recap',
      training_stage: 'stage_progress',
      inputs: { model: 'original-model' },
      parameters: { steps: 5000, batch_size: 16 },
      gpu_count: 2,
      requested_gpu_uuids: ['GPU-a', 'GPU-b'],
      auto_config: {
        model_type: 'pi05',
        finetune_mode: 'full',
        action_dim: 32,
        action_horizon: 10,
      },
      dataset_bindings: [
        {
          dataset_asset_id: dataset.id,
          norm_asset_id: norm.id,
          weight: 1,
          processor: {
            name: 'ComposableDataConfig',
            parameters: {
              composition: {
                stage_progress_key: 'progress.labeled',
                aux_keys: ['stage_progress_gt'],
              },
            },
          },
        },
      ],
      stage_config: { n_step: 50, positive_ratio: 0.3, stage_count: 2 },
      attempts: [{ id: 'attempt', state: 'SUCCEEDED' }],
      outputs: [checkpoint],
    };
    const catalogCheckpoint = {
      ...checkpoint,
      stage_annotation_source: {
        run_id: run.id,
        run_name: run.name,
        attempt_id: checkpoint.origin_attempt_id,
        step: 4999,
      },
    };
    return {
      run,
      checkpoint,
      catalogCheckpoint,
      dataset,
      norm,
      assets: [catalogCheckpoint, dataset, norm],
    };
  };

  await test('annotation picker trusts catalog provenance and labels the source task and step', () => {
    const { checkpoint, catalogCheckpoint } = fixture();
    assert.equal(recapModelError(catalogCheckpoint, 'stage_annotation'), undefined);
    assert.equal(stageAnnotationCheckpointLabel(catalogCheckpoint), '原阶段进度 · 第 4,999 步');
    assert.equal(recapModelError(checkpoint, 'stage_progress'), undefined);
    for (const rejected of [
      checkpoint,
      {
        ...checkpoint,
        metadata: {
          ...checkpoint.metadata,
          stage_annotation_source: catalogCheckpoint.stage_annotation_source,
        },
      },
      { ...catalogCheckpoint, origin_attempt_id: 'different-attempt' },
      { ...catalogCheckpoint, state: 'REGISTERED' },
      { ...catalogCheckpoint, state: 'ARCHIVED' },
      { ...catalogCheckpoint, metadata: { ...checkpoint.metadata, availability: 'removed' } },
    ])
      assert.match(recapModelError(rejected, 'stage_annotation'), /已成功完成/);
  });

  await test('Stage Advantage annotation accepts registered PyTorch checkpoints with compatible roles', () => {
    const external = {
      id: 'external',
      node_id: 'node',
      name: '外部阶段进度模型',
      version: 'v1',
      kind: 'MODEL',
      origin_attempt_id: null,
      state: 'REGISTERED',
      metadata: { format: 'openpi-pytorch' },
    };
    assert.equal(recapModelError(external, 'stage_annotation'), undefined);
    assert.equal(stageAnnotationCheckpointLabel(external), '外部阶段进度模型 · v1');
    assert.equal(
      recapModelError(
        {
          ...external,
          state: 'READY',
          metadata: { format: 'torch', model_role: 'stage_progress_estimator' },
        },
        'stage_annotation',
      ),
      undefined,
    );
    for (const rejected of [
      { ...external, metadata: { format: 'openpi-orbax' } },
      { ...external, metadata: undefined },
      { ...external, metadata: { ...external.metadata, model_role: 'value_estimator' } },
      { ...external, metadata: { ...external.metadata, model_role: 'policy' } },
      { ...external, metadata: { ...external.metadata, availability: 'removed' } },
      { ...external, state: 'ARCHIVED' },
      { ...external, kind: 'NORM' },
      { ...external, origin_attempt_id: 'unverified-attempt' },
    ])
      assert.match(recapModelError(rejected, 'stage_annotation'), /PyTorch/);
    assert.match(recapModelError(external, 'value_annotation'), /Value 训练/);
  });

  await test('annotation handoff preserves composition and checkpoint while dropping training controls', () => {
    const { run, checkpoint, assets } = fixture();
    const snapshot = stageAdvantageHandoffSnapshot(
      run,
      { stage: 'stage_annotation', checkpoint },
      assets,
    );
    const { request } = snapshot;
    assert.equal(request.training_stage, 'stage_annotation');
    assert.deepEqual(request.inputs, { model: checkpoint.id });
    assert.deepEqual(request.dataset_bindings, run.dataset_bindings);
    assert.deepEqual(request.auto_config, run.auto_config);
    assert.deepEqual(request.parameters, { batch_size: 16 });
    assert.deepEqual(request.stage_config, run.stage_config);
    assert.equal(request.gpu_count, 1);
    assert.equal(request.requested_gpu_uuids, undefined);
    assert.equal(
      decodeTrainingDraft(encodeTrainingDraft(snapshot)).request.training_stage,
      'stage_annotation',
    );
    assert.deepEqual(retryTrainingSnapshot({ ...run, ...request }).request.inputs, request.inputs);
  });

  await test('managed Value annotation freezes one chosen source row and the checkpoint-bound norm', () => {
    const { run, checkpoint, dataset, norm } = fixture();
    run.training_stage = 'value';
    run.stage_config = { val_ratio: 0.2 };
    run.dataset_bindings.push({ ...run.dataset_bindings[0], dataset_asset_id: 'data-2' });
    const shared = { ...norm, id: 'shared' };
    Object.assign(checkpoint, {
      norm_asset_id: shared.id,
      metadata: { format: 'openpi-pytorch', model_role: 'value_estimator' },
      value_annotation_source: {
        run_id: run.id,
        run_name: run.name,
        attempt_id: 'attempt',
        step: 50,
      },
    });
    const handoff = { stage: 'value_annotation', checkpoint, sourceIndex: 1 };
    const { request } = recapValueAnnotationSnapshot(run, handoff, [
      checkpoint,
      dataset,
      { ...dataset, id: 'data-2' },
      norm,
      shared,
    ]);
    assert.equal(request.training_stage, 'value_annotation');
    assert.deepEqual(request.inputs, { model: checkpoint.id });
    assert.equal(request.dataset_bindings.length, 1);
    assert.equal(request.dataset_bindings[0].dataset_asset_id, 'data-2');
    assert.equal(request.dataset_bindings[0].norm_asset_id, shared.id);
    assert.deepEqual(request.parameters, { batch_size: 16 });
    assert.equal(request.gpu_count, 1);
    assert.equal(request.stage_config, undefined);
  });

  await test('continue with prior estimator weights creates a fresh train-only job', () => {
    const { run, checkpoint, dataset, norm } = fixture();
    const { request } = stageAdvantageHandoffSnapshot(
      run,
      { stage: 'stage_progress', checkpoint },
      [checkpoint, dataset, norm],
    );
    assert.deepEqual(request.inputs, { model: checkpoint.id });
    assert.equal(request.training_stage, 'stage_progress');
    assert.deepEqual(request.parameters, run.parameters);
    assert.equal(request.stage_config, undefined);
    assert.equal(request.resume, undefined);
    assert.equal(run.inputs.model, 'original-model');
  });

  await test('progress handoff requires same-node assets and a successful estimator output', () => {
    const { run, checkpoint, catalogCheckpoint, dataset, norm, assets } = fixture();
    assert.deepEqual(stageAdvantageCheckpoints(run), [checkpoint]);
    const handoff = { stage: 'stage_annotation', checkpoint };
    assert.throws(
      () =>
        stageAdvantageHandoffSnapshot(run, handoff, [
          catalogCheckpoint,
          dataset,
          { ...norm, node_id: 'other' },
        ]),
      /归一化参数在当前机器不可用/,
    );
    assert.throws(
      () =>
        stageAdvantageHandoffSnapshot(
          { ...run, attempts: [{ id: 'attempt', state: 'FAILED' }] },
          handoff,
          assets,
        ),
      /可用的模型检查点/,
    );
    assert.throws(
      () => stageAdvantageHandoffSnapshot(run, handoff, [checkpoint, dataset, norm]),
      /有效标注来源记录/,
    );
    const other = {
      ...catalogCheckpoint,
      stage_annotation_source: {
        ...catalogCheckpoint.stage_annotation_source,
        run_id: 'different-run',
      },
    };
    assert.throws(
      () => stageAdvantageHandoffSnapshot(run, handoff, [other, dataset, norm]),
      /有效标注来源记录/,
    );
    assert.throws(
      () =>
        stageAdvantageHandoffSnapshot(run, handoff, [
          { ...checkpoint, metadata: { format: 'openpi-pytorch' } },
          dataset,
          norm,
        ]),
      /可用的模型检查点/,
    );
  });
} finally {
  await server.close();
}
