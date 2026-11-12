import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

const outputDir = '/runs/Value run';
const makeAsset = (id, kind, path, overrides = {}) => ({
  id,
  kind,
  name: id,
  path,
  node_id: 'node-a',
  state: 'READY',
  version: '1',
  ...overrides,
});
const sourceA = makeAsset('a', 'DATASET', '/datasets/a');
const sourceB = makeAsset('b', 'DATASET', "/datasets/robot's set");
const checkpoint = makeAsset('value-checkpoint', 'MODEL', `${outputDir}/checkpoints/20`, {
  origin_attempt_id: 'successful-attempt',
  metadata: { format: 'openpi-pytorch', model_role: 'value_estimator', step: 20 },
});

function valueRun(overrides = {}) {
  return {
    id: 'value-run',
    name: 'Value training',
    node_id: 'node-a',
    profile_id: 'openpi-umi',
    template_id: 'recap',
    state: 'SUCCEEDED',
    training_stage: 'value',
    created_at: '2026-11-12T10:00:00Z',
    dataset_bindings: [sourceA, sourceB].map((asset) => ({
      dataset_asset_id: asset.id,
      norm_asset_id: 'norm',
      weight: 1,
      recipe_index: 0,
    })),
    attempts: [
      { id: 'failed-attempt', state: 'FAILED' },
      {
        id: 'successful-attempt',
        state: 'SUCCEEDED',
        output_dir: outputDir,
        code_snapshot: { code_dir: "/repos/operator's OpenPI" },
      },
    ],
    outputs: [checkpoint],
    ...overrides,
  };
}

try {
  const { recapAnnotationInputs, recapValueCheckpoints, recapAnnotationCommand } =
    await server.ssrLoadModule('/src/recapAnnotation.ts');

  await test('annotation sources preserve config row order and original indices for missing rows', () => {
    const run = valueRun({ inputs: { dataset_0: sourceB.id, dataset_1: sourceA.id } });
    assert.deepEqual(recapAnnotationInputs(run, [sourceB, sourceA]), [
      { asset: sourceA, index: 0 },
      { asset: sourceB, index: 1 },
    ]);
    assert.deepEqual(recapAnnotationInputs(run, [sourceB]), [{ asset: sourceB, index: 1 }]);
    assert.deepEqual(recapAnnotationInputs(run, [{ ...sourceA, state: 'ARCHIVED' }, sourceB]), [
      { asset: sourceB, index: 1 },
    ]);
  });

  await test('legacy source bindings use numeric dataset roles and the unnumbered single dataset', () => {
    const run = valueRun({
      dataset_bindings: null,
      inputs: { dataset_11: sourceB.id, dataset_2: { id: sourceA.id }, norm_2: 'norm' },
    });
    assert.deepEqual(recapAnnotationInputs(run, [sourceB, sourceA]), [
      { asset: sourceA, index: 2 },
      { asset: sourceB, index: 11 },
    ]);
    assert.deepEqual(
      recapAnnotationInputs({ ...run, inputs: { dataset: sourceB.id } }, [sourceA, sourceB]),
      [{ asset: sourceB, index: 0 }],
    );
  });

  await test('Value checkpoints are sorted and restricted to ready same-machine successful attempts', () => {
    const legacy = {
      ...checkpoint,
      id: 'legacy',
      version: '100',
      metadata: { format: 'openpi-orbax' },
    };
    const run = valueRun({
      outputs: [
        checkpoint,
        { ...checkpoint, id: 'removed', metadata: { availability: 'removed' } },
        { ...checkpoint, id: 'archived', state: 'ARCHIVED' },
        { ...checkpoint, id: 'policy', metadata: { model_role: 'policy' } },
        { ...checkpoint, id: 'other-node', node_id: 'node-b' },
        { ...checkpoint, id: 'failed', origin_attempt_id: 'failed-attempt' },
        { ...checkpoint, id: 'unlinked', origin_attempt_id: null },
        legacy,
      ],
    });
    assert.deepEqual(recapValueCheckpoints(run), [legacy, checkpoint]);
    assert.deepEqual(recapValueCheckpoints({ ...run, training_stage: 'offline_rl' }), []);
    assert.deepEqual(
      recapAnnotationInputs({ ...run, training_stage: 'offline_rl' }, [sourceB]),
      [],
    );
  });

  await test('PyTorch annotation command quotes shell paths and keeps the actual source row index', () => {
    const command = recapAnnotationCommand(
      valueRun(),
      checkpoint,
      { asset: sourceB, index: 1 },
      "/outputs/copy's $(ignored)",
    );
    assert.equal(
      command,
      [
        "cd '/repos/operator'\\''s OpenPI' &&",
        'uv run python scripts/acp/lerobot_value_infer_pytorch.py',
        "  --config-file '/runs/Value run/training_config.py'",
        "  --checkpoint-dir '/runs/Value run/checkpoints/20'",
        "  --dataset-root '/datasets/robot'\\''s set'",
        "  --output-dir '/outputs/copy'\\''s $(ignored)'",
        '  --dataset-index 1',
      ].join(' \\\n'),
    );
  });

  await test('annotation command rejects JAX, wrong bindings, and paths overlapping the original data', () => {
    const run = valueRun();
    const source = { asset: sourceB, index: 1 };
    const output = '/outputs/annotated';
    const jax = { ...checkpoint, metadata: { format: 'openpi-orbax' } };
    assert.equal(
      recapAnnotationCommand(valueRun({ outputs: [jax] }), jax, source, output),
      undefined,
    );
    assert.equal(
      recapAnnotationCommand(run, checkpoint, { ...source, index: 0 }, output),
      undefined,
    );
    assert.equal(
      recapAnnotationCommand(run, { ...checkpoint, id: 'unknown' }, source, output),
      undefined,
    );
    for (const bad of [
      'relative/output',
      sourceB.path,
      `${sourceB.path}/./child`,
      '/datasets/',
      '/outputs/../datasets/child',
    ]) {
      assert.equal(recapAnnotationCommand(run, checkpoint, source, bad), undefined, bad);
    }
    assert.match(
      recapAnnotationCommand(run, checkpoint, source, `${sourceB.path}-annotated`),
      /--dataset-index 1$/u,
    );
  });
} finally {
  await server.close();
}
