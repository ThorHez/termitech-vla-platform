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
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { ApiError } = await server.ssrLoadModule('/src/api.ts');
  const {
    StorageConfiguration,
    checkedStorageOperation,
    storageDirectoryError,
    storageSettings,
    storageConfigurationError,
    storageConfigurationSupported,
  } = await server.ssrLoadModule('/src/StorageConfiguration.tsx');
  const node = {
    id: 'machine-a', name: '训练机器', status: 'ONLINE',
    output_root: '/data/outputs', cache_root: null,
    metrics: { capabilities: ['STORAGE_CONFIGURE'] },
  };
  const render = (props = {}, permissions = ['machines.manage']) =>
    renderToStaticMarkup(React.createElement(AuthContext.Provider, {
      value: { can: (permission) => permissions.includes(permission) },
    }, React.createElement(StorageConfiguration, { node, ...props })));
  const button = (html) => html.match(/<button\b[^>]*>/)?.[0] || '';
  const ready = () => ({
    id: 'storage-op-a', node_id: node.id, status: 'READY',
    requested: { output_root: '/new/outputs', cache_root: '/new/cache' },
    storage: {
      output_root: '/new/outputs', cache_root: '/new/cache',
      directories: {
        output: { path: '/new/outputs', free_bytes: 4 * 2 ** 40, total_bytes: 6 * 2 ** 40 },
        cache: { path: '/new/cache', free_bytes: 4 * 2 ** 40, total_bytes: 6 * 2 ** 40 },
      },
    },
    error: null,
  });

  await test('only machine managers receive the storage settings entry', () => {
    assert.match(render(), /修改存储目录/);
    assert.doesNotMatch(button(render()), /disabled/);
    for (const permissions of [[], ['platform.read'], ['training.write'], ['assets.write']])
      assert.equal(render({}, permissions), '');
  });

  await test('busy and offline machines disable storage edits with actionable reasons', () => {
    const busy = render({ inUse: true });
    assert.match(button(busy), /disabled=""/);
    assert.match(busy, /未结束的训练任务/);
    const offline = render({ node: { ...node, status: 'OFFLINE' } });
    assert.match(button(offline), /disabled=""/);
    assert.match(offline, /机器在线后/);
  });

  await test('explicit capabilities gate edits while legacy capability omission remains compatible', () => {
    for (const capabilities of [[], ['PROFILE_REBIND'], null, 'STORAGE_CONFIGURE']) {
      const old = { ...node, metrics: { capabilities } };
      assert.equal(storageConfigurationSupported(old), false);
      const html = render({ node: old });
      assert.match(button(html), /disabled=""/);
      assert.match(html, /请更新 Agent 后重试/);
    }
    for (const metrics of [undefined, {}, { capabilities: ['STORAGE_CONFIGURE'] }])
      assert.equal(storageConfigurationSupported({ ...node, metrics }), true);
  });

  await test('directory validation rejects traversal, root and control characters while cache may use defaults', () => {
    for (const path of ['', '/', '///', 'relative/path', '/data/../cache', '/data/./cache', '/data/\u0000x', '/data\\x'])
      assert.equal(typeof storageDirectoryError(path), 'string', JSON.stringify(path));
    assert.equal(storageDirectoryError(''), storageDirectoryError('   '));
    assert.equal(storageDirectoryError('   ', true), undefined);
    assert.equal(storageDirectoryError('/data3/new folder/cache/'), undefined);
    assert.deepEqual(storageSettings(' /data3//outputs/// ', '  '), {
      output_root: '/data3/outputs', cache_root: null,
    });
    assert.deepEqual(storageSettings('/data3/outputs', '/data4//cache/'), {
      output_root: '/data3/outputs', cache_root: '/data4/cache',
    });
  });

  await test('a successful receipt must bind to the current machine, operation and exact requested directories', () => {
    const value = ready();
    assert.equal(checkedStorageOperation(value, node, value.id), value);
    const edits = [
      (row) => { row.node_id = 'other-machine'; },
      (row) => { row.id = 'other-operation'; },
      (row) => { row.storage.output_root = '/unrequested/output'; },
      (row) => { row.storage.directories.cache.path = '/unrequested/cache'; },
      (row) => { row.storage.directories.output = null; },
      (row) => { row.storage = null; },
      (row) => { row.requested.output_root = '/'; },
      (row) => { row.requested.cache_root = ''; },
      (row) => { row.status = 'APPLIED'; },
    ];
    for (const edit of edits) {
      const row = structuredClone(value); edit(row);
      assert.throws(() => checkedStorageOperation(row, node, value.id), /回执与当前机器或请求不一致/);
    }
  });

  await test('disk capacity evidence rejects invalid values and default caches require a null directory receipt', () => {
    for (const free of [-1, NaN, Infinity, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
      const row = ready(); row.storage.directories.output.free_bytes = free;
      assert.throws(() => checkedStorageOperation(row, node));
    }
    const row = ready();
    row.storage.directories.output.free_bytes = row.storage.directories.output.total_bytes + 1;
    assert.throws(() => checkedStorageOperation(row, node));
    const defaults = ready();
    defaults.requested.cache_root = defaults.storage.cache_root = null;
    assert.throws(() => checkedStorageOperation(defaults, node));
    defaults.storage.directories.cache = null;
    assert.equal(checkedStorageOperation(defaults, node), defaults);
  });

  await test('pending and failed receipts do not claim that directory configuration has been persisted', () => {
    for (const status of ['PENDING', 'FAILED']) {
      const row = { ...ready(), status, storage: null };
      assert.equal(checkedStorageOperation(row, node), row);
    }
  });

  await test('unsupported connection errors are readable while conflicts retain server detail', () => {
    for (const code of ['READ_ONLY_REAL_AGENT', 'AGENT_STORAGE_CONFIGURE_UNSUPPORTED', 'COMMAND_UNSUPPORTED'])
      assert.equal(storageConfigurationError(new ApiError(code, 'raw technical error', 409)).message,
        '当前连接尚不支持修改存储目录，请更新 Agent 后重试。');
    const conflict = new ApiError('STORAGE_CONFIGURE_CONFLICT', '存储设置已变化，请刷新后重试', 409);
    assert.equal(storageConfigurationError(conflict), conflict);
  });
} finally {
  await server.close();
}
