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

const binding = (dataset_asset_id, weight) => ({
  dataset_asset_id,
  norm_asset_id: `norm-${dataset_asset_id}`,
  weight,
  recipe_index: 0,
});
const ready = (count) => ({ status: 'ready', count });
const near = (actual, expected) => {
  assert.equal(typeof actual, 'number');
  assert.ok(Number.isFinite(actual));
  assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} should approximate ${expected}`);
};
const freeze = (value) => {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
};

try {
  const { datasetSamplingShares, samplingPercentText } =
    await server.ssrLoadModule('/src/datasetSampling.ts');

  await test('weight, source sample and expected sampling shares are three different ratios', () => {
    const rows = datasetSamplingShares([binding('a', 4), binding('b', 1)], {
      a: ready(100),
      b: ready(900),
    });
    assert.deepEqual(
      rows.map((row) => row.sampleCount),
      [100, 900],
    );
    assert.deepEqual(
      rows.map((row) => row.weightPercent),
      [80, 20],
    );
    assert.deepEqual(
      rows.map((row) => row.samplePercent),
      [10, 90],
    );
    near(rows[0].samplingPercent, (4 / 13) * 100);
    near(rows[1].samplingPercent, (9 / 13) * 100);
    assert.ok(rows.every((row) => !row.weightIssue && !row.sampleIssue && !row.samplingIssue));
    assert.equal(samplingPercentText(rows[0].samplingPercent), '约 30.77%');
    assert.equal(samplingPercentText(rows[1].samplingPercent), '约 69.23%');
  });

  await test('equal per-sample weights retain source-size sampling rather than equal source probability', () => {
    const rows = datasetSamplingShares([binding('a', 1), binding('b', 1)], {
      a: ready(100),
      b: ready(900),
    });
    assert.deepEqual(
      rows.map((row) => row.weightPercent),
      [50, 50],
    );
    assert.deepEqual(
      rows.map((row) => row.samplePercent),
      [10, 90],
    );
    assert.deepEqual(
      rows.map((row) => row.samplingPercent),
      [10, 90],
    );
  });

  await test('each binding row contributes its own source mass even when identities repeat', () => {
    const rows = datasetSamplingShares(
      [binding('same-source', 1), binding('same-source', 3), binding('other-source', 2)],
      { 'same-source': ready(100), 'other-source': ready(200) },
    );
    assert.deepEqual(
      rows.map((row) => row.sampleCount),
      [100, 100, 200],
    );
    assert.deepEqual(
      rows.map((row) => row.samplePercent),
      [25, 25, 50],
    );
    near(rows[0].weightPercent, 100 / 6);
    near(rows[1].weightPercent, 50);
    near(rows[2].weightPercent, 100 / 3);
    near(rows[0].samplingPercent, 12.5);
    near(rows[1].samplingPercent, 37.5);
    near(rows[2].samplingPercent, 50);
  });

  await test('a missing, loading or unavailable count never renormalizes known sources to 100 percent', () => {
    for (const reading of [
      undefined,
      { status: 'loading' },
      { status: 'loading', count: 900 },
      { status: 'unavailable', count: 900 },
    ]) {
      const rows = datasetSamplingShares([binding('a', 4), binding('b', 1)], {
        a: ready(100),
        ...(reading === undefined ? {} : { b: reading }),
      });
      assert.deepEqual(
        rows.map((row) => row.weightPercent),
        [80, 20],
      );
      assert.deepEqual(
        rows.map((row) => row.sampleCount),
        [100, undefined],
      );
      for (const row of rows) {
        assert.equal(row.samplePercent, undefined);
        assert.equal(row.samplingPercent, undefined);
        assert.match(row.sampleIssue, /第 2 组/);
        assert.match(row.samplingIssue, /第 2 组/);
      }
    }
  });

  await test('an unselected dataset and malformed frame counts remain unknown', () => {
    const empty = datasetSamplingShares([binding('', 4), binding('b', 1)], {
      '': ready(100),
      b: ready(900),
    });
    assert.equal(empty[0].sampleCount, undefined);
    assert.equal(empty[1].samplePercent, undefined);
    for (const count of [
      undefined,
      null,
      '100',
      false,
      -1,
      1.5,
      NaN,
      Infinity,
      Number.MAX_SAFE_INTEGER + 1,
    ]) {
      const rows = datasetSamplingShares([binding('a', 4), binding('b', 1)], {
        a: ready(count),
        b: ready(900),
      });
      assert.equal(rows[0].sampleCount, undefined);
      assert.ok(
        rows.every((row) => row.samplePercent === undefined && row.samplingPercent === undefined),
      );
      assert.deepEqual(
        rows.map((row) => row.weightPercent),
        [80, 20],
      );
    }
  });

  await test('observed zero counts retain zero mass but an all-empty mixture cannot divide by zero', () => {
    const mixed = datasetSamplingShares([binding('a', 4), binding('b', 1)], {
      a: ready(0),
      b: ready(900),
    });
    assert.deepEqual(
      mixed.map((row) => row.sampleCount),
      [0, 900],
    );
    assert.deepEqual(
      mixed.map((row) => row.samplePercent),
      [0, 100],
    );
    assert.deepEqual(
      mixed.map((row) => row.samplingPercent),
      [0, 100],
    );
    assert.equal(samplingPercentText(mixed[0].samplingPercent), '约 0.00%');
    const empty = datasetSamplingShares([binding('a', 4), binding('b', 1)], {
      a: ready(0),
      b: ready(0),
    });
    assert.deepEqual(
      empty.map((row) => row.sampleCount),
      [0, 0],
    );
    assert.deepEqual(
      empty.map((row) => row.weightPercent),
      [80, 20],
    );
    for (const row of empty) {
      assert.equal(row.samplePercent, undefined);
      assert.equal(row.samplingPercent, undefined);
      assert.equal(row.sampleIssue, '样本总数为 0');
      assert.equal(row.samplingIssue, '样本总数为 0');
    }
  });

  await test('invalid weights block weighted ratios without hiding valid original sample shares', () => {
    for (const weight of [NaN, Infinity, -Infinity, 0, -1, '4']) {
      const rows = datasetSamplingShares([binding('a', weight), binding('b', 1)], {
        a: ready(100),
        b: ready(900),
      });
      assert.deepEqual(
        rows.map((row) => row.sampleCount),
        [100, 900],
      );
      assert.deepEqual(
        rows.map((row) => row.samplePercent),
        [10, 90],
      );
      for (const row of rows) {
        assert.equal(row.weightPercent, undefined);
        assert.equal(row.samplingPercent, undefined);
        assert.match(row.weightIssue, /有效正权重/);
        assert.match(row.samplingIssue, /有效正权重/);
        assert.equal(row.sampleIssue, undefined);
      }
    }
  });

  await test('large finite weights and safe frame counts avoid overflow in sums and products', () => {
    const equal = datasetSamplingShares([binding('a', 1e308), binding('b', 1e308)], {
      a: ready(9e15),
      b: ready(3e15),
    });
    assert.deepEqual(
      equal.map((row) => row.weightPercent),
      [50, 50],
    );
    near(equal[0].samplePercent, 75);
    near(equal[1].samplePercent, 25);
    near(equal[0].samplingPercent, 75);
    near(equal[1].samplingPercent, 25);
    const different = datasetSamplingShares([binding('a', 1e308), binding('b', 5e307)], {
      a: ready(9e15),
      b: ready(3e15),
    });
    near(different[0].weightPercent, (2 / 3) * 100);
    near(different[1].weightPercent, (1 / 3) * 100);
    near(different[0].samplePercent, 75);
    near(different[1].samplePercent, 25);
    near(different[0].samplingPercent, (6 / 7) * 100);
    near(different[1].samplingPercent, (1 / 7) * 100);
  });

  await test('ratio calculation never changes bindings or account-local count readings', () => {
    const bindings = freeze([binding('a', 4), binding('b', 1)]);
    const counts = freeze({ a: ready(100), b: ready(900) });
    const before = structuredClone({ bindings, counts });
    const rows = datasetSamplingShares(bindings, counts);
    rows[0].sampleCount = 500;
    rows[0].weightPercent = 0;
    assert.deepEqual({ bindings, counts }, before);
  });

  await test('empty lists and unavailable text never invent a numeric percentage', () => {
    assert.deepEqual(datasetSamplingShares([], {}), []);
    assert.equal(samplingPercentText(undefined), '暂无法计算');
    assert.equal(
      samplingPercentText(undefined, '缺少第 2 组样本数'),
      '暂无法计算（缺少第 2 组样本数）',
    );
    assert.equal(samplingPercentText(80), '约 80.00%');
  });

  await test('dataset cards render all three ratios from verified source frame counts', async () => {
    const { DatasetBindings } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
    const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
    const node = { id: 'node', name: 'Machine', status: 'ONLINE', gpus: [] };
    const dataset = (id, frames) => ({
      id,
      node_id: node.id,
      name: `Dataset ${id}`,
      kind: 'DATASET',
      path: `/datasets/${id}`,
      version: 'v1',
      state: 'REGISTERED',
      validation: {
        status: 'SUCCEEDED',
        scope: 'dataset_source',
        completed_at: '2026-11-04T00:00:00Z',
        timestamp_contract: { status: 'verified', checked_rows: frames },
      },
    });
    const html = renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { user: { id: 'reader' }, loading: false, expired: false } },
        React.createElement(DatasetBindings, {
          bindings: [binding('a', 4), binding('b', 1)],
          slots: [],
          assets: [dataset('a', 100), dataset('b', 900)],
          node,
          showProcessing: false,
          readOnly: true,
          onChange() {},
        }),
      ),
    );
    const cards = [...html.matchAll(/<div class="dataset-binding-overview">([\s\S]*?)<\/div>/g)];
    assert.equal(cards.length, 2);
    assert.match(cards[0][1], /权重占比约 80\.00%/);
    assert.match(cards[0][1], /样本数：100/);
    assert.match(cards[0][1], /样本数占比约 10\.00%/);
    assert.match(cards[0][1], /预计采样占比：约 30\.77%/);
    assert.match(cards[1][1], /权重占比约 20\.00%/);
    assert.match(cards[1][1], /样本数：900/);
    assert.match(cards[1][1], /样本数占比约 90\.00%/);
    assert.match(cards[1][1], /预计采样占比：约 69\.23%/);
  });
} finally {
  await server.close();
}
