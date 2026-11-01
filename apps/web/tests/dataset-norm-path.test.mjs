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
const node = { id: 'source', name: 'source', status: 'ONLINE', hostname: 'source', roots: ['/data'] };
const norm = { id: 'norm', node_id: 'source', name: 'original norm', kind: 'NORM', version: 'v1', path: '/data/original', state: 'REGISTERED' };
const assets = [norm, { ...norm, id: 'foreign', node_id: 'other' }, { ...norm, id: 'archived', state: 'ARCHIVED' }];

try {
  const { DATASET_NORM_PATH, datasetNormBindingPayload, datasetNormInputError, datasetNormPathError } =
    await server.ssrLoadModule('/src/datasetNormAssets.ts');
  const { DatasetNormField } = await server.ssrLoadModule('/src/DatasetNormBinding.tsx');
  const html = (props) => renderToStaticMarkup(React.createElement(DatasetNormField, {
    nodeId: node.id, node, assets, value: '', onChange() {}, pathValue: '', onPathChange() {}, ...props,
  }));

  await test('path input creates one atomic norm_path request rather than an asset-id and path combination', () => {
    const payload = datasetNormBindingPayload(DATASET_NORM_PATH, ' /data/norm_stats.json ');
    assert.deepEqual(payload, { norm_path: '/data/norm_stats.json' });
    assert.equal(Object.hasOwn(payload, 'norm_asset_id'), false);
    assert.equal(Object.hasOwn(payload, 'node_id'), false);
    assert.equal(Object.hasOwn(payload, 'path'), false);
  });
  await test('existing binding and explicit clearing retain their original API shape', () => {
    assert.deepEqual(datasetNormBindingPayload('norm', '/data/ignored'), { norm_asset_id: 'norm' });
    assert.deepEqual(datasetNormBindingPayload(''), { norm_asset_id: null });
  });
  await test('directory and norm_stats.json paths are both supported', () => {
    for (const path of ['/data/normalization', '/data/normalization/', '/data/normalization/norm_stats.json', '/data/中文 norm/norm_stats.json']) {
      assert.equal(datasetNormPathError(path), undefined, path);
      assert.equal(datasetNormInputError('source', DATASET_NORM_PATH, path, assets), undefined);
    }
  });
  await test('empty, relative, parent traversal, control characters and root paths cannot be submitted', () => {
    for (const path of ['', ' ', 'data/stats', '../norm', '/data/../norm', '/data\\norm', '/data/\0stats', '/data/\nstats', '\n/data/stats', '/data/stats\n', '/', '//', '/./', '/data/' + 'x'.repeat(4096)]) {
      assert.ok(datasetNormPathError(path), JSON.stringify(path));
    }
    assert.match(datasetNormInputError('', DATASET_NORM_PATH, '/data/stats', assets), /先选择/);
  });
  await test('path mode ignores an unavailable old selection while normal selection remains node-scoped', () => {
    assert.equal(datasetNormInputError('source', DATASET_NORM_PATH, '/data/new', assets), undefined);
    assert.match(datasetNormInputError('source', 'archived', '/data/new', assets), /归档或不可用/);
    assert.match(datasetNormInputError('source', 'foreign', '/data/new', assets), /同一台机器/);
    assert.equal(datasetNormInputError('source', '', '', assets), undefined);
  });
  await test('path entry is an explicit selector choice and does not add a separate norm registration form', () => {
    const rendered = html({});
    assert.match(rendered, /直接填写路径（自动登记）/);
    assert.doesNotMatch(rendered, /norm_stats 绝对路径/);
    assert.doesNotMatch(rendered, /value="foreign"|value="archived"/);
  });
  await test('path mode renders a required autocomplete and explains registration versus file validation', () => {
    const rendered = html({ value: DATASET_NORM_PATH, pathValue: '/data/new/norm_stats.json' });
    assert.match(rendered, /norm_stats 绝对路径/);
    assert.match(rendered, /value="\/data\/new\/norm_stats\.json"/);
    assert.match(rendered, /保存时自动登记并绑定/);
    assert.match(rendered, /已有登记会复用/);
    assert.match(rendered, /训练预检中验证/);
    assert.doesNotMatch(rendered, /已绑定的 norm_stats 资产不存在/);
  });
  await test('offline machines permit manual path input and parent busy state disables both controls', () => {
    const offline = html({ value: DATASET_NORM_PATH, pathValue: '/data/norm', node: { ...node, status: 'OFFLINE' } });
    assert.match(offline, /norm_stats 绝对路径/);
    assert.doesNotMatch(offline, /<input[^>]+disabled=""/);
    const busy = html({ value: DATASET_NORM_PATH, pathValue: '/data/norm', disabled: true });
    assert.match(busy, /<select[^>]+disabled=""/);
    assert.match(busy, /<input[^>]+disabled=""/);
  });
  await test('invalid path remains visible with an error and cannot masquerade as an unavailable asset ID', () => {
    const rendered = html({ value: DATASET_NORM_PATH, pathValue: '/data/../bad' });
    assert.match(rendered, /路径不能包含/);
    assert.doesNotMatch(rendered, /__path__ · 不可用/);
  });
  await test('legacy component callers without path props keep the original selection-only behavior', () => {
    const rendered = html({ onPathChange: undefined, value: 'norm' });
    assert.match(rendered, /value="norm" selected=""/);
    assert.match(rendered, /\/data\/original/);
    assert.doesNotMatch(rendered, /直接填写路径/);
  });
} finally {
  await server.close();
}
