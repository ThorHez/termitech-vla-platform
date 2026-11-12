import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const fixtures = JSON.parse(
  readFileSync(new URL('../../../tests/fixtures/composed-processing-v1.json', import.meta.url)),
);
const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const { recapAcpSnapshot } = await server.ssrLoadModule('/src/recapAcp.ts');
  const { encodeTrainingDraft, decodeTrainingDraft } =
    await server.ssrLoadModule('/src/trainingDraft.ts');
  const fixture = () => {
    const norm = { id: 'norm', kind: 'NORM', node_id: 'node', state: 'READY' };
    const dataset = {
      id: 'annotated',
      kind: 'DATASET',
      node_id: 'node',
      state: 'REGISTERED',
      metadata: {
        recap_annotation_source: {
          run_id: 'run',
          attempt_id: 'attempt',
          checkpoint_asset_id: 'checkpoint',
          source_dataset_asset_id: 'source',
          source_norm_asset_id: norm.id,
        },
      },
    };
    const checkpoint = {
      id: 'checkpoint',
      kind: 'MODEL',
      node_id: 'node',
      state: 'ARCHIVED',
      origin_attempt_id: 'attempt',
    };
    const run = {
      id: 'run',
      template_id: 'recap',
      name: '折衣 Value',
      training_stage: 'value',
      state: 'SUCCEEDED',
      node_id: 'node',
      profile_id: 'profile',
      recipe: 'value_recipe',
      stage_config: { val_ratio: 0.1 },
      parameters: { steps: 40000, batch_size: 128, ema_decay: 0.99 },
      requested_gpu_uuids: ['GPU-source'],
      inputs: { model: 'value_model' },
      attempts: [{ id: 'attempt', state: 'SUCCEEDED' }],
      outputs: [checkpoint],
      dataset_bindings: [
        {
          dataset_asset_id: 'source',
          norm_asset_id: norm.id,
          weight: 5,
          processor: {
            name: 'ComposableDataConfig',
            parameters: {
              composition: { ...structuredClone(fixtures.valid[0]), aux_keys: ['value_target'] },
            },
          },
        },
      ],
    };
    return { run, dataset, norm };
  };

  await test('an attached Value annotation seeds ACP with fresh controls and no initial model', () => {
    const { run, dataset, norm } = fixture();
    const snapshot = recapAcpSnapshot(run, dataset, [dataset, norm]);
    assert.equal(snapshot.step, 0);
    assert.equal(snapshot.advancedRecipe, false);
    assert.equal(snapshot.request.training_stage, 'offline_rl');
    assert.equal(snapshot.request.node_id, run.node_id);
    assert.equal(snapshot.request.profile_id, run.profile_id);
    assert.deepEqual(snapshot.request.inputs, {});
    assert.deepEqual(snapshot.request.parameters, {});
    for (const key of ['recipe', 'stage_config', 'entrypoint', 'requested_gpu_uuids']) {
      assert.equal(Object.hasOwn(snapshot.request, key), false, key);
    }
    const [binding] = snapshot.request.dataset_bindings;
    assert.equal(binding.dataset_asset_id, dataset.id);
    assert.equal(binding.norm_asset_id, norm.id);
    assert.equal(binding.weight, 1);
    assert.deepEqual(binding.processor.parameters.composition.aux_keys, ['is_positive']);
    assert.deepEqual(run.dataset_bindings[0].processor.parameters.composition.aux_keys, [
      'value_target',
    ]);
    assert.deepEqual(decodeTrainingDraft(encodeTrainingDraft(snapshot)), snapshot);
  });

  await test('Kai0 ACP handoff requires a registered output from the successful attempt', () => {
    const { run, dataset, norm } = fixture();
    run.training_stage = 'stage_progress';
    dataset.origin_attempt_id = 'attempt';
    dataset.metadata = { source_dataset_asset_id: 'source', source_norm_asset_id: norm.id };
    assert.throws(() => recapAcpSnapshot(run, dataset, [dataset, norm]), /来源与当前训练不一致/);
    run.outputs.push(dataset);
    assert.equal(
      recapAcpSnapshot(run, dataset, [dataset, norm]).request.training_stage,
      'offline_rl',
    );
  });

  await test('annotation lineage picks the exact source row when a dataset is bound twice', () => {
    const { run, dataset, norm } = fixture();
    const second = structuredClone(run.dataset_bindings[0]);
    second.processor.parameters.composition.normalization = 'zscore';
    run.dataset_bindings.push(second);
    run.inputs.dataset_2 = 'source';
    run.inputs.norm_2 = norm.id;
    dataset.metadata.recap_annotation_source.source_dataset_index = 1;
    const { request } = recapAcpSnapshot(run, dataset, [dataset, norm]);
    assert.equal(
      request.dataset_bindings[0].processor.parameters.composition.normalization,
      'zscore',
    );
    dataset.metadata.recap_annotation_source.source_dataset_index = 2;
    assert.throws(
      () => recapAcpSnapshot(run, dataset, [dataset, norm]),
      /原始数据和归一化参数与来源训练不一致/,
    );
  });

  await test('stale dataset and mismatched norm or source cannot replace the ACP draft', () => {
    const { run, dataset, norm } = fixture();
    assert.throws(() => recapAcpSnapshot(run, dataset, [norm]), /标注数据集当前不可用/);
    assert.throws(
      () => recapAcpSnapshot(run, dataset, [{ ...dataset, state: 'ARCHIVED' }, norm]),
      /标注数据集当前不可用/,
    );
    assert.throws(
      () => recapAcpSnapshot(run, dataset, [dataset, { ...norm, node_id: 'other' }]),
      /归一化参数在当前机器不可用/,
    );
    dataset.metadata.recap_annotation_source.source_dataset_asset_id = 'other';
    assert.throws(
      () => recapAcpSnapshot(run, dataset, [dataset, norm]),
      /原始数据和归一化参数与来源训练不一致/,
    );
  });
} finally {
  await server.close();
}
