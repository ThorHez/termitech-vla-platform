import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { middlewareMode: true, hmr: false, ws: false }, optimizeDeps: { noDiscovery: true, include: [] }, appType: 'custom' });
const profile = { schema_version: 1, format: 'lerobot', robot_type: 'ARX5', arm_mode: 'single', action_representation: 'unknown', action_mode: 'unknown', rotation_representation: 'unknown', task_description: 'Fold the cloth', cameras: [{ name: 'top', key: 'observation.images.top', kind: 'rgb' }], field_annotations: [{ key: 'actions', role: 'action', representation: 'composite', value_mode: 'delta', component: 'other', arm: 'left', shape: [16, 10], horizon: 16 }] };
const entry = (path, changes = {}) => ({ path, relative_path: path.replace('/data/', ''), name: path.split('/').at(-1), features: [{ key: 'observation.images.top', dtype: 'video', shape: [3, 224, 224], names: [] }], summary: {}, error: null, ...changes });
const scan = (entries) => ({ id: 'scan', node_id: 'source', path: '/data', recursive: true, max_depth: 4, state: 'READY', entries, warnings: [], truncated: false, scanned_directories: 10, skipped_directories: 0, error: null, expires_at: new Date(Date.now() + 1800000).toISOString() });
try {
  const { scanRows, applyBatchNorm, batchRegistrationEntries, batchRegistrationError, batchIdempotency, batchScanPathError, canonicalDatasetPath, datasetScanScope } = await server.ssrLoadModule('/src/batchDatasetRegistrationData.ts');
  const { BatchDatasetRegistration } = await server.ssrLoadModule('/src/BatchDatasetRegistration.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { DATASET_NORM_PATH } = await server.ssrLoadModule('/src/datasetNormAssets.ts');
  const goodRows = () => scanRows(scan([entry('/data/cloth-a'), entry('/data/cloth-b')]), []);
  const args = (changes = {}) => ({ nodeId: 'source', rows: goodRows(), version: 'v1', metadataOnly: true, profile, assets: [], ...changes });

  await test('canonical scope normalizes trailing slashes without allowing parent traversal', () => {
    assert.equal(canonicalDatasetPath('/data//./cloth/'), '/data/cloth');
    assert.equal(canonicalDatasetPath('/data/../secret'), undefined);
    assert.equal(datasetScanScope('source', ' /data/ ', true, 4), datasetScanScope('source', '/data', true, 4));
    assert.notEqual(datasetScanScope('other', '/data', true, 4), datasetScanScope('source', '/data', true, 4));
    for (const path of ['', ' ', '/', '/./', 'relative', '/data/../private', '/data\\private', '/data/\0bad']) assert.ok(batchScanPathError(path));
  });
  await test('scan selects only new valid datasets, preserving archived and unavailable duplicate authority', () => {
    const rows = scanRows(scan([entry('/data/new'), entry('/data/known', { existing_asset_id: 'known', existing_state: 'ARCHIVED' }), entry('/data/invalid', { error: { code: 'BAD_META', message: 'Invalid metadata' } }), entry('/data/cached/')]), [{ id: 'cached', node_id: 'source', kind: 'DATASET', path: '/data/cached', state: 'UNAVAILABLE' }]);
    assert.deepEqual(rows.map((row) => row.selected), [true, false, false, false]);
    assert.equal(rows[3].existingAssetId, 'cached');
    assert.equal(scanRows(scan([entry('/data/new')]), [{ node_id: 'other', kind: 'DATASET', path: '/data/new' }])[0].selected, true);
  });
  await test('duplicate folder basenames receive readable relative-path names', () => {
    const rows = scanRows(scan([entry('/data/day-1/base'), entry('/data/day-2/base')]), []);
    assert.deepEqual(rows.map((row) => row.name), ['day-1 · base', 'day-2 · base']);
  });
  await test('invalid or repeated paths and over-limit scan responses are not silently accepted', () => {
    assert.throws(() => scanRows(scan([entry('/data/a'), entry('/data/a/')]), []));
    assert.throws(() => scanRows(scan([entry('/data/../private')]), []));
    assert.throws(() => scanRows(scan(Array.from({ length: 201 }, (_, i) => entry(`/data/${i}`))), []));
  });
  await test('metadata-only registration copies exact discovered paths and no invented profile', () => {
    const rows = goodRows(); rows[1].selected = false;
    assert.deepEqual(batchRegistrationEntries(rows, ' v2 ', true, profile), [{ path: '/data/cloth-a', name: 'cloth-a', version: 'v2', norm_asset_id: null }]);
  });
  await test('shared template retains composite labels and original horizon without training composition or norm identity', () => {
    const rows = batchRegistrationEntries(goodRows(), 'v1', false, { ...profile, processing: { malformed: true } });
    assert.equal(rows.length, 2); assert.equal(rows[0].profile.field_annotations[0].representation, 'composite');
    assert.equal(rows[0].profile.field_annotations[0].horizon, 16); assert.equal(Object.hasOwn(rows[0].profile, 'processing'), false);
    rows[0].profile.field_annotations[0].shape[0] = 1;
    assert.equal(rows[1].profile.field_annotations[0].shape[0], 16); assert.equal(profile.field_annotations[0].shape[0], 16);
  });
  await test('explicit shared norm path produces one path field per selected row without asset ID ambiguity', () => {
    const rows = batchRegistrationEntries(applyBatchNorm(goodRows(), DATASET_NORM_PATH, ' /data/stats/norm_stats.json '), 'v1', true, profile);
    for (const row of rows) { assert.equal(row.norm_path, '/data/stats/norm_stats.json'); assert.equal(Object.hasOwn(row, 'norm_asset_id'), false); }
  });
  await test('empty selection, invalid names, invalid shared profiles and foreign norm bindings block registration', () => {
    assert.equal(batchRegistrationError(args()), undefined);
    assert.match(batchRegistrationError(args({ rows: goodRows().map((row) => ({ ...row, selected: false })) })), /至少/);
    assert.match(batchRegistrationError(args({ rows: goodRows().map((row) => ({ ...row, name: '' })) })), /名称/);
    assert.match(batchRegistrationError(args({ metadataOnly: false, profile: { ...profile, robot_type: '' } })), /公共数据集格式/);
    assert.match(batchRegistrationError(args({ rows: applyBatchNorm(goodRows(), 'foreign', ''), assets: [{ id: 'foreign', node_id: 'other', kind: 'NORM' }] })), /cloth-a.*同一台机器/);
    assert.match(batchRegistrationError(args({ rows: applyBatchNorm(goodRows(), DATASET_NORM_PATH, '../secret') })), /cloth-a.*绝对路径/);
  });
  await test('each dataset independently chooses an existing norm, its own path, or no binding', () => {
    const rows = scanRows(scan([entry('/data/cloth-a'), entry('/data/cloth-b'), entry('/data/cloth-c')]), []);
    assert.ok(rows.every((row) => row.normId === '' && row.normPath === ''));
    rows[0].normId = 'stats-a';
    rows[1].normId = DATASET_NORM_PATH;
    rows[1].normPath = ' /data/cloth-b/norm_stats.json ';
    const assets = [{ id: 'stats-a', node_id: 'source', kind: 'NORM', state: 'REGISTERED' }];
    assert.equal(batchRegistrationError(args({ rows, assets })), undefined);
    const entries = batchRegistrationEntries(rows, 'v1', true, profile);
    assert.equal(entries[0].norm_asset_id, 'stats-a');
    assert.deepEqual(Object.keys(entries[1]).filter((key) => key.startsWith('norm_')), ['norm_path']);
    assert.equal(entries[1].norm_path, '/data/cloth-b/norm_stats.json');
    assert.equal(entries[2].norm_asset_id, null);
    const identity = batchIdempotency(undefined, entries, () => 'first');
    rows[1].normPath = '/data/cloth-b/updated_stats';
    assert.equal(batchIdempotency(identity, batchRegistrationEntries(rows, 'v1', true, profile), () => 'changed').key, 'changed');
  });
  await test('common norm applies only on demand to selected eligible rows; later edits stay independent', () => {
    const rows = scanRows(scan([entry('/data/a'), entry('/data/b'), entry('/data/c'), entry('/data/known', { existing_asset_id: 'known' }), entry('/data/bad', { error: { code: 'BAD', message: 'bad metadata' } })]), []);
    rows[2].selected = false;
    rows[2].normId = DATASET_NORM_PATH; rows[2].normPath = '/data/c/stats';
    rows[3].selected = true; rows[4].selected = true;
    const changed = applyBatchNorm(rows, 'shared', '');
    assert.deepEqual(changed.map((row) => row.normId), ['shared', 'shared', DATASET_NORM_PATH, '', '']);
    assert.equal(rows[0].normId, '');
    changed[0].normId = DATASET_NORM_PATH; changed[0].normPath = '/data/a/stats';
    assert.equal(changed[1].normId, 'shared');
    const cleared = applyBatchNorm(changed, '', '');
    assert.deepEqual(cleared.slice(0, 3).map((row) => [row.normId, row.normPath]), [['', ''], ['', ''], [DATASET_NORM_PATH, '/data/c/stats']]);
  });
  await test('unselected unfinished norm paths do not block or leak into a batch; reselecting shows the dataset error', () => {
    const rows = goodRows();
    rows[1].normId = DATASET_NORM_PATH; rows[1].normPath = ''; rows[1].selected = false;
    assert.equal(batchRegistrationError(args({ rows })), undefined);
    assert.equal(batchRegistrationEntries(rows, 'v1', true, profile).length, 1);
    rows[1].selected = true;
    assert.match(batchRegistrationError(args({ rows })), /cloth-b.*完整路径/);
    rows[1].normPath = '/data/cloth-b/stats';
    assert.equal(batchRegistrationError(args({ rows })), undefined);
  });
  await test('an unchanged network retry reuses its idempotency key; editing selection gets a new key', () => {
    let created = 0; const key = () => `key-${++created}`;
    const first = batchIdempotency(undefined, { scan_id: 'scan', entries: goodRows() }, key);
    assert.equal(batchIdempotency(first, { scan_id: 'scan', entries: goodRows() }, key), first);
    const edited = batchIdempotency(first, { scan_id: 'scan', entries: goodRows().slice(0, 1) }, key);
    assert.equal(edited.key, 'key-2');
  });
  await test('a large shared format prompts smaller batches before sending an oversized request', () => {
    const manyRows = scanRows(scan(Array.from({ length: 100 }, (_, index) => entry(`/data/dataset-${index}`))), []);
    const large = { ...profile, feature_keys: ['actions', ...Array.from({ length: 150 }, (_, index) => `observation.${index}.` + 'x'.repeat(150))] };
    assert.match(batchRegistrationError(args({ rows: manyRows, metadataOnly: false, profile: large })), /1 MiB/);
    assert.equal(batchRegistrationError(args({ rows: manyRows.slice(0, 2), metadataOnly: false, profile: large })), undefined);
  });
  await test('viewer and legacy contexts cannot directly mount the private batch workflow', () => {
    for (const auth of [{ can: () => false }, { can: () => true }]) {
      const html = renderToStaticMarkup(React.createElement(AuthContext.Provider, { value: auth }, React.createElement(BatchDatasetRegistration, { close() {} })));
      assert.match(html, /具有数据集登记权限的个人账户/);
      assert.doesNotMatch(html, /扫描数据集/);
    }
  });
  await test('dataset registrars see the scan stage without public asset edit permission', () => {
    const html = renderToStaticMarkup(React.createElement(AuthContext.Provider, { value: { user: { id: 'operator' }, can: (p) => p === 'datasets.register' } }, React.createElement(DataProvider, { enabled: false }, React.createElement(BatchDatasetRegistration, { close() {} }))));
    assert.match(html, /批量登记数据集/); assert.match(html, /数据集父目录/); assert.match(html, /扫描数据集/);
    assert.doesNotMatch(html, /登记 0 个数据集/);
  });
  await test('asset edit permission alone does not bypass the dataset registration gate', () => {
    const html = renderToStaticMarkup(React.createElement(AuthContext.Provider, { value: { user: { id: 'custom-writer' }, can: (p) => p === 'assets.write' } }, React.createElement(BatchDatasetRegistration, { close() {} })));
    assert.match(html, /具有数据集登记权限的个人账户/);
    assert.doesNotMatch(html, /扫描数据集/);
  });
} finally { await server.close(); }
