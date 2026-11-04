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

const node = {
  id: 'source',
  name: 'T001host',
  hostname: 'source-host',
  status: 'ONLINE',
  connection_address: { host: '192.0.2.10', port: 22, source: 'ssh' },
};
const norm = {
  id: 'norm',
  kind: 'NORM',
  node_id: node.id,
  name: 'cloth raw14',
  path: '/data/cloth/norm_stats_raw14/norm_stats.json',
  version: 'v2',
  status: 'REGISTERED',
};
const dataset = {
  id: 'dataset',
  kind: 'DATASET',
  node_id: node.id,
  name: 'fold_cloth',
  path: '/data/a-very-long-workspace/cloth/demonstrations/complete-dataset-path',
  version: 'v3',
  status: 'REGISTERED',
  norm_asset_id: norm.id,
  created_at: '2026-11-01T01:02:03.000000+00:00',
  metadata: {
    dataset_profile: {
      schema_version: 1,
      format: 'lerobot',
      robot_type: 'ARX5',
      arm_mode: 'bimanual',
      action_representation: 'joint',
      action_mode: 'absolute',
      rotation_representation: 'none',
      cameras: [{ name: 'top_head', key: 'observation.images.top_head', kind: 'rgb' }],
      task_description: 'Flatten and fold the cloth.',
      fps: 30,
      feature_keys: ['observation.state', 'action'],
      field_annotations: [],
    },
  },
};

const buttons = (html) =>
  [...html.matchAll(/<button\b([^>]*)>([\s\S]*?)<\/button>/g)].map((match) => ({
    attributes: match[1],
    label: match[2].replace(/<[^>]*>/g, '').trim(),
  }));
const button = (html, pattern) => buttons(html).find((value) => pattern.test(value.label));
const normAction = /^(?:绑定|修改) norm_stats$/;
const mutation =
  /^(?:(?:绑定|修改) norm_stats|(?:编辑|补录)数据特征|同步到其他机器|删除(?:登记|资产)?(?:登记)?)$/;

try {
  const { AssetDetail } = await server.ssrLoadModule('/src/AssetDetail.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const render = (
    asset = dataset,
    source = node,
    role = 'admin',
    assets = [dataset, norm],
    props = {},
  ) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: {
            can: (permission) =>
              role === 'admin'
                ? ['assets.write', 'assets.transfer', 'dataset-templates.write'].includes(
                    permission,
                  )
                : role === 'operator' &&
                  ['assets.transfer', 'dataset-templates.write'].includes(permission),
          },
        },
        React.createElement(AssetDetail, {
          asset,
          node: source,
          assets,
          close() {},
          editProfile() {},
          bindNorm() {},
          sync() {},
          remove() {},
          ...props,
        }),
      ),
    );

  await test('asset details show complete paths, machine identity, version and norm binding', () => {
    const html = render();
    assert.match(html, /role="dialog"/);
    assert.match(html, /fold_cloth/);
    assert.match(html, /T001host/);
    assert.match(html, /v3/);
    assert.ok(html.includes(dataset.path), 'the full dataset path must remain readable');
    assert.ok(html.includes(norm.path), 'the bound norm full path must remain readable');
    assert.match(html, /cloth raw14/);
    assert.match(html, /href="#\/nodes\/source"/);
  });

  await test('dataset details retain registered profile, task and camera field information', () => {
    const html = render();
    assert.match(html, /ARX5/);
    assert.match(html, /双臂/);
    assert.match(html, /Flatten and fold the cloth\./);
    assert.match(html, /observation\.images\.top_head/);
    assert.match(html, /30/);
  });

  await test('bound norm state and repair entry remain visible inside dataset details', () => {
    for (const [status, label] of [
      ['REGISTERED', '待校验'],
      ['UNAVAILABLE', 'UNAVAILABLE'],
      ['ARCHIVED', '已归档'],
    ]) {
      const html = render(dataset, node, 'admin', [dataset, { ...norm, status }]);
      assert.match(html, new RegExp(`aria-label="norm_stats 状态"[\\s\\S]*?${label}`));
      assert.ok(html.includes(norm.name));
      assert.ok(html.includes(norm.path));
      assert.ok(html.includes(norm.version));
      assert.ok(button(html, normAction));
    }
  });

  await test('each dataset sharing a norm retains its own details and a descriptively named dataset picker', () => {
    const second = { ...dataset, id: 'second', name: 'second dataset' };
    for (const current of [dataset, second]) {
      const html = render(current, node, 'admin', [dataset, second, norm], {
        datasetChoices: [dataset, second],
        selectDataset() {},
      });
      assert.match(html, /aria-label="使用此归一化参数的数据集"/);
      assert.doesNotMatch(html, /集合中的数据集/);
      assert.ok(html.includes(norm.path));
      assert.ok(button(html, normAction));
    }
  });

  await test('administrators can maintain public datasets while operators can only synchronize them', () => {
    for (const role of ['admin']) {
      const html = render(dataset, node, role);
      for (const label of [normAction, /数据特征/, /同步到其他机器/, /删除/]) {
        assert.ok(button(html, label), `missing operation ${label} for ${role}`);
      }
      assert.doesNotMatch(button(html, /同步到其他机器/).attributes, /\bdisabled/);
    }
    const operator = render(dataset, node, 'operator');
    assert.ok(button(operator, /同步到其他机器/));
    for (const label of [normAction, /数据特征/, /删除/])
      assert.equal(button(operator, label), undefined);
  });

  await test('viewers can inspect every detail without being offered mutations', () => {
    const html = render(dataset, node, 'viewer');
    assert.ok(html.includes(dataset.path));
    assert.ok(html.includes(norm.path));
    assert.match(html, /Flatten and fold the cloth\./);
    assert.equal(buttons(html).filter((value) => mutation.test(value.label)).length, 0);
  });

  await test('offline source machines keep details readable while disabling only synchronization', () => {
    const html = render(dataset, { ...node, status: 'OFFLINE' });
    assert.match(button(html, /同步到其他机器/).attributes, /\bdisabled/);
    assert.ok(button(html, normAction));
    assert.ok(button(html, /数据特征/));
    assert.ok(button(html, /删除/));
    assert.ok(html.includes(dataset.path));
  });

  await test('unavailable datasets can repair metadata but cannot bind norms or synchronize', () => {
    const html = render({ ...dataset, status: 'UNAVAILABLE' });
    assert.equal(button(html, normAction), undefined);
    assert.ok(button(html, /数据特征/));
    assert.match(button(html, /同步到其他机器/).attributes, /\bdisabled/);
    assert.ok(button(html, /删除/));
  });

  await test('archived assets expose historical information without mutation controls', () => {
    for (const asset of [
      { ...dataset, status: 'ARCHIVED', state: 'REGISTERED' },
      { ...dataset, status: undefined, state: 'ARCHIVED' },
    ]) {
      const html = render(asset);
      assert.ok(html.includes(dataset.path));
      assert.equal(buttons(html).filter((value) => mutation.test(value.label)).length, 0);
    }
  });

  await test('model and norm details keep shared operations without dataset-only controls', () => {
    for (const kind of ['MODEL', 'NORM']) {
      const html = render({ ...norm, kind, metadata: undefined, norm_asset_id: undefined });
      assert.ok(button(html, /同步到其他机器/));
      assert.ok(button(html, /删除/));
      assert.equal(button(html, normAction), undefined);
      assert.equal(button(html, /数据特征/), undefined);
    }
  });

  await test('model details show the checkpoint-bound norm without dataset mutation controls', () => {
    const model = { ...dataset, kind: 'MODEL', name: 'Folding model', metadata: {} };
    const html = render(model, node, 'admin', [model, norm]);
    assert.match(html, /模型训练归一化参数/);
    assert.match(html, /随此模型 checkpoint 保存，供推理使用/);
    assert.ok(html.includes(norm.path));
    assert.ok(html.includes(norm.name));
    assert.equal(button(html, normAction), undefined);
    assert.equal(button(html, /数据特征/), undefined);
  });

  await test('missing machine and dataset metadata do not prevent inspecting an asset', () => {
    const html = render({ ...dataset, metadata: undefined, norm_asset_id: null }, null);
    assert.ok(html.includes(dataset.path));
    assert.ok(button(html, /数据特征/));
    assert.match(button(html, /同步到其他机器/).attributes, /\bdisabled/);
  });

  await test('source-provided names, paths and task descriptions are rendered as text', () => {
    const html = render({
      ...dataset,
      name: '<img src=x onerror=alert(1)>',
      path: '/data/<script>alert(1)</script>',
      metadata: {
        dataset_profile: {
          ...dataset.metadata.dataset_profile,
          task_description: '<svg onload=alert(1)>fold</svg>',
        },
      },
    });
    assert.match(html, /&lt;img/);
    assert.match(html, /&lt;script&gt;/);
    assert.match(html, /&lt;svg/);
    assert.doesNotMatch(html, /<img\b|<script\b|<svg\s+onload=/);
  });
} finally {
  await server.close();
}
