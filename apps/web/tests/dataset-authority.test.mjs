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
  const { recipeDatasetSlots, newDatasetBindings } = await server.ssrLoadModule(
    '/src/datasetBindingHelpers.ts',
  );
  const { DatasetBindings } = await server.ssrLoadModule('/src/DatasetBindings.tsx');
  await test('recipe paths and mixing weights never silently become task inputs', () => {
    const detail = {
      fields: [
        { path: 'data', kind: 'expression', expression: 'MultiDataConfigFactory(...)' },
        { path: 'data.datasets[0]', kind: 'expression', expression: 'UmiProcessing(...)' },
        { path: 'data.datasets[0].repo_id', kind: 'literal', value: '/source/dataset' },
        { path: 'data.datasets[0].assets.assets_dir', kind: 'literal', value: '/source/norm' },
        { path: 'data.datasets[0].assets.asset_id', kind: 'literal', value: '.' },
        { path: 'data.datasets[1]', kind: 'expression', expression: 'OtherProcessing(...)' },
        { path: 'data.weights', kind: 'literal', value: [1, 50] },
      ],
    };
    const slots = recipeDatasetSlots(detail);
    assert.deepEqual(
      slots.map((row) => row.processing),
      ['UmiProcessing', 'OtherProcessing'],
    );
    assert.equal(JSON.stringify(slots).includes('/source/'), false);
    assert.deepEqual(
      newDatasetBindings(slots),
      [0, 1].map((recipe_index) => ({
        recipe_index,
        weight: 1,
        dataset_asset_id: '',
        norm_asset_id: '',
      })),
    );
    assert.deepEqual(newDatasetBindings([]), [
      { recipe_index: 0, weight: 1, dataset_asset_id: '', norm_asset_id: '' },
    ]);
  });
  await test('training inputs show registered assets and independent editable task weights', () => {
    const html = renderToStaticMarkup(
      React.createElement(DatasetBindings, {
        bindings: [
          {
            recipe_index: 0,
            weight: 20,
            dataset_asset_id: 'registered-data',
            norm_asset_id: 'registered-norm',
          },
        ],
        slots: [{ recipe_index: 0, weight: 1, processing: 'UmiProcessing', multiple: true }],
        assets: [
          {
            id: 'registered-data',
            kind: 'DATASET',
            name: 'Dataset',
            version: 'v1',
            path: '/registered/data',
          },
          {
            id: 'registered-norm',
            kind: 'NORM',
            name: 'Norm',
            version: 'v1',
            path: '/registered/norm/norm_stats.json',
          },
        ],
        onChange() {},
      }),
    );
    assert.match(html, /本次训练只使用这里选择的已登记资产/);
    assert.match(html, /value="registered-data" selected/);
    assert.match(html, /value="registered-norm" selected/);
    assert.match(html, /type="number"[^>]+value="20"/);
    assert.match(html, /\/registered\/data/);
  });
} finally {
  await server.close();
}
