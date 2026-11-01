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

const run = (changes = {}) => ({
  id: 'same-id',
  name: '叠衣训练',
  state: 'RUNNING',
  node_id: 'source',
  profile_id: 'openpi',
  template_id: 'openpi-sft',
  created_at: '2026-10-31T01:00:00Z',
  parameters: { steps: 1000 },
  attempts: [{ id: 'attempt', latest_metrics: { step: 200 } }],
  ...changes,
});
const transfer = (changes = {}) => ({
  id: 'same-id',
  name: '叠衣数据同步',
  state: 'TRANSFERRING',
  source_asset_id: 'data',
  source_kind: 'DATASET',
  source_node_id: 'source',
  source_path: '/data/original-cloth',
  target_node_id: 'target',
  target_directory: '/data/imports',
  target_name: 'cloth-copy',
  target_path: '/data/imports/cloth-copy',
  bytes_transferred: 256,
  bytes_total: 1024,
  files_transferred: 1,
  files_total: 4,
  speed_bytes_per_second: 128,
  progress_pct: 25,
  created_at: '2026-10-31T02:00:00Z',
  updated_at: '2026-10-31T02:00:10Z',
  ...changes,
});
const nodes = [
  { id: 'source', name: 'T001host', hostname: 'source-workstation', status: 'ONLINE' },
  { id: 'target', name: 'Thor', hostname: 'target-jetson', status: 'ONLINE' },
];

try {
  const {
    buildBackgroundTasks,
    filterBackgroundTasks,
    backgroundTaskPath,
    backgroundTaskProgress,
    backgroundTaskActive,
    backgroundTaskActiveCounts,
  } = await server.ssrLoadModule('/src/backgroundTaskEntries.ts');
  const { BackgroundTaskTable, BackgroundTaskErrors, BackgroundTasksPage } =
    await server.ssrLoadModule('/src/BackgroundTasks.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const trainingTask = (changes = {}) => buildBackgroundTasks([run(changes)], [])[0];
  const syncTask = (changes = {}) => buildBackgroundTasks([], [transfer(changes)])[0];

  await test('sidebar active total includes the same training and sync tasks as the active list', () => {
    const runs = [run({ state: 'RUNNING' })];
    const transfers = [transfer({ state: 'TRANSFERRING' })];
    assert.deepEqual(backgroundTaskActiveCounts(runs, transfers), {
      training: 1,
      sync: 1,
      total: 2,
    });
    assert.equal(
      backgroundTaskActiveCounts(runs, transfers).total,
      filterBackgroundTasks(buildBackgroundTasks(runs, transfers), { status: 'ACTIVE' }).length,
    );
  });
  await test('active badge excludes terminal records and updates when sync finishes or stops', () => {
    const runs = [run({ state: 'RUNNING' }), run({ id: 'done', state: 'SUCCEEDED' })];
    for (const state of ['SUCCEEDED', 'FAILED', 'CANCELLED', 'INTERRUPTED']) {
      assert.deepEqual(backgroundTaskActiveCounts(runs, [transfer({ state })]), {
        training: 1,
        sync: 0,
        total: 1,
      });
    }
    assert.deepEqual(backgroundTaskActiveCounts([], []), { training: 0, sync: 0, total: 0 });
    const active = transfer({ state: 'VERIFYING' });
    assert.equal(backgroundTaskActiveCounts(runs, [active, active]).total, 2);
  });

  await test('training and sync records retain distinct identities and sort by actual creation time', () => {
    const tasks = buildBackgroundTasks(
      [run(), run({ id: 'latest', created_at: '2026-10-31T11:00:00+08:00' })],
      [transfer()],
    );
    assert.deepEqual(
      tasks.map((task) => task.key),
      ['training:latest', 'data-sync:same-id', 'training:same-id'],
    );
    assert.equal(backgroundTaskPath(tasks[1]), '/runs/sync/same-id');
    assert.equal(backgroundTaskPath(tasks[2]), '/runs/same-id');
    assert.equal(backgroundTaskPath(syncTask({ id: 'task/id' })), '/runs/sync/task%2Fid');
  });

  await test('duplicate receipts cannot duplicate a task while invalid dates remain stable at the end', () => {
    const tasks = buildBackgroundTasks(
      [run(), run(), run({ id: 'legacy', created_at: 'invalid' })],
      [transfer(), transfer()],
    );
    assert.deepEqual(
      tasks.map((task) => task.key),
      ['data-sync:same-id', 'training:same-id', 'training:legacy'],
    );
  });

  await test('type and status filters operate on both task kinds without calling interrupted sync active', () => {
    const tasks = buildBackgroundTasks(
      [run()],
      [
        transfer(),
        transfer({ id: 'broken', state: 'INTERRUPTED' }),
        transfer({ id: 'done', state: 'SUCCEEDED' }),
      ],
    );
    assert.equal(filterBackgroundTasks(tasks, { type: 'training' }).length, 1);
    assert.equal(filterBackgroundTasks(tasks, { type: 'data-sync' }).length, 3);
    assert.equal(filterBackgroundTasks(tasks, { status: 'ACTIVE' }).length, 2);
    assert.deepEqual(
      filterBackgroundTasks(tasks, { status: 'FAILED' }).map((task) => task.item.id),
      ['broken'],
    );
    assert.equal(filterBackgroundTasks(tasks, { status: 'SUCCEEDED' }).length, 1);
    for (const state of [
      'QUEUED',
      'PREPARING',
      'SCANNING',
      'TRANSFERRING',
      'VERIFYING',
      'CANCELLING',
    ])
      assert.equal(backgroundTaskActive(syncTask({ state })), true);
    for (const state of ['FAILED', 'INTERRUPTED', 'CANCELLED', 'SUCCEEDED'])
      assert.equal(backgroundTaskActive(syncTask({ state })), false);
    assert.equal(
      filterBackgroundTasks([syncTask({ state: 'CANCELLED' })], { status: 'CANCELLED' }).length,
      1,
    );
  });

  await test('search finds task names, both machines, hostnames and copied paths', () => {
    const tasks = buildBackgroundTasks([run()], [transfer()]);
    assert.equal(filterBackgroundTasks(tasks, { nodes, query: ' THOR ' }).length, 1);
    assert.equal(filterBackgroundTasks(tasks, { nodes, query: 't001HOST' }).length, 2);
    assert.equal(filterBackgroundTasks(tasks, { nodes, query: 'target-jetson' }).length, 1);
    assert.equal(filterBackgroundTasks(tasks, { nodes, query: 'original-cloth' }).length, 1);
    assert.equal(filterBackgroundTasks(tasks, { nodes, query: 'cloth-copy' }).length, 1);
    assert.equal(filterBackgroundTasks(tasks, { nodes, query: '叠衣' }).length, 2);
    assert.equal(filterBackgroundTasks(tasks, { nodes, query: 'no-match' }).length, 0);
  });

  await test('preparation and missing telemetry stay indeterminate instead of inventing zero progress', () => {
    for (const state of ['QUEUED', 'PREPARING', 'SCANNING']) {
      const progress = backgroundTaskProgress(syncTask({ state, progress_pct: 99 }));
      assert.equal(progress.percent, null);
      assert.equal(progress.preparing, true);
    }
    const unknown = backgroundTaskProgress(
      syncTask({ progress_pct: null, bytes_total: 0, bytes_transferred: 0 }),
    );
    assert.equal(unknown.percent, null);
    assert.equal(unknown.label, '进度待更新');
    assert.equal(backgroundTaskProgress(trainingTask({ attempts: [] })).percent, null);
  });

  await test('transfer progress and speed use actual receipts, with 100 reserved for completed tasks', () => {
    const progress = backgroundTaskProgress(syncTask());
    assert.equal(progress.percent, 25);
    assert.match(progress.detail, /256 B \/ 1 KiB.*128 B\/s/);
    assert.equal(backgroundTaskProgress(syncTask({ progress_pct: null })).percent, 25);
    assert.equal(backgroundTaskProgress(syncTask({ progress_pct: 120 })).percent, 99);
    assert.equal(backgroundTaskProgress(syncTask({ state: 'SUCCEEDED' })).percent, 100);
    assert.doesNotMatch(backgroundTaskProgress(syncTask({ state: 'FAILED' })).detail, /\/s/);
    const training = backgroundTaskProgress(trainingTask());
    assert.equal(training.percent, 20);
    assert.equal(training.label, '200 / 1,000 步');
  });

  await test('task table exposes task type, source and target, proper routes and meaningful progress', () => {
    const html = renderToStaticMarkup(
      React.createElement(BackgroundTaskTable, {
        tasks: buildBackgroundTasks([run()], [transfer()]),
        nodes,
        templates: [{ id: 'openpi-sft', name: 'OpenPI / UMI' }],
      }),
    );
    assert.match(html, /任务类型/);
    assert.match(html, /模型训练/);
    assert.match(html, /数据同步/);
    assert.match(html, /href="#\/runs\/same-id"/);
    assert.match(html, /href="#\/runs\/sync\/same-id"/);
    assert.match(html, /Thor/);
    assert.match(html, /T001host/);
    assert.match(html, /OpenPI \/ UMI/);
    assert.match(html, /25\.0%/);
    assert.match(html, /200 \/ 1,000 步/);
    assert.doesNotMatch(html, /取消同步|重试同步/);
  });

  await test('an error from one task source stays separate and never hides records from the other', () => {
    const html = renderToStaticMarkup(
      React.createElement(
        React.Fragment,
        null,
        React.createElement(BackgroundTaskErrors, {
          transferError: new Error('SSH status unavailable'),
          refreshTraining() {},
          refreshTransfers() {},
        }),
        React.createElement(BackgroundTaskTable, {
          tasks: buildBackgroundTasks([run()], []),
          nodes,
        }),
      ),
    );
    assert.match(html, /数据同步任务读取失败/);
    assert.match(html, /SSH status unavailable/);
    assert.match(html, /叠衣训练/);
    assert.doesNotMatch(html, /训练任务读取失败/);
  });

  await test('page keeps the training entry and adds task-type filtering without transfer writes', () => {
    const html = renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: () => true } },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(BackgroundTasksPage, { createRun() {} }),
        ),
      ),
    );
    assert.match(html, /训练任务/);
    assert.match(html, /新建训练/);
    assert.match(html, /后台执行/);
    assert.match(html, /筛选任务类型/);
    assert.match(html, /最近 200 条/);
    assert.doesNotMatch(html, /开始同步|取消同步|重试同步/);
  });
} finally {
  await server.close();
}
