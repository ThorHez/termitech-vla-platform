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
  const { TransferCard } = await server.ssrLoadModule('/src/CheckpointTransfers.tsx');
  const { transferElapsed } = await server.ssrLoadModule('/src/checkpointTransfer.ts');
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const base = {
    id: 'transfer-preparing',
    name: '准备中的同步',
    source_node_id: 'source',
    target_node_id: 'target',
    target_directory: '/target/models',
    target_path: null,
    state: 'EXPORTING',
    bytes_total: 0,
    bytes_transferred: 0,
    files_total: 0,
    files_transferred: 0,
    speed_bytes_per_second: 0,
    progress_pct: 0,
    attempt: 1,
    logs: '',
    created_at: '2026-10-27T10:00:00Z',
    updated_at: '2026-10-27T10:01:24Z',
  };
  const render = (item) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        {
          value: { can: () => false },
        },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(TransferCard, { item, changed() {} }),
        ),
      ),
    );

  await test('unknown totals show preparation stages without fabricated byte, file, or percentage progress', () => {
    for (const [state, label] of [
      ['QUEUED', '等待同步'],
      ['EXPORTING', '准备推理配置'],
      ['PREPARING', '检查机器与目录'],
    ]) {
      for (const bytes_total of [0, -1]) {
        const html = render({ ...base, state, bytes_total });
        assert.match(html, new RegExp(`${label}，尚未开始传输`));
        assert.match(html, /checkpoint-transfer-indeterminate/);
        assert.match(html, /role="progressbar"/);
        assert.match(html, /文件总大小与数量待统计/);
        assert.doesNotMatch(html, /aria-valuenow|0\.0%|0 B|个文件/);
      }
    }
  });

  await test('failure before byte transfer preserves the real error without a misleading zero progress bar', () => {
    const html = render({
      ...base,
      state: 'FAILED',
      finished_at: '2026-10-27T10:01:24Z',
      error: { code: 'SSH_CREDENTIAL_DECRYPT_FAILED', message: '源机器 SSH 凭据解密失败' },
    });
    assert.match(html, /传输尚未开始/);
    assert.match(html, /SSH_CREDENTIAL_DECRYPT_FAILED/);
    assert.match(html, /源机器 SSH 凭据解密失败/);
    assert.match(html, /role="alert"/);
    assert.match(html, /任务耗时 1分24秒/);
    assert.doesNotMatch(
      html,
      /role="progressbar"|0\.0%|0 B|个文件|checkpoint-transfer-indeterminate/,
    );
  });

  await test('known byte totals retain real percentage and counts while success remains 100 percent', () => {
    const known = {
      ...base,
      state: 'TRANSFERRING',
      bytes_total: 128 * 1024 ** 2,
      bytes_transferred: 64 * 1024 ** 2,
      files_total: 10,
      files_transferred: 5,
      progress_pct: 50,
    };
    const html = render(known);
    assert.match(html, /aria-valuenow="50"/);
    assert.match(html, /50\.0%/);
    assert.match(html, /64 MiB[^]*128 MiB/);
    assert.match(html, /5[^]*10[^]*个文件/);
    assert.doesNotMatch(html, /尚未开始传输|总大小与数量待统计|checkpoint-transfer-indeterminate/);
    for (const bytes_total of [0, 128 * 1024 ** 2]) {
      const finished = render({
        ...known,
        state: 'SUCCEEDED',
        bytes_total,
        finished_at: '2026-10-27T10:01:24Z',
      });
      assert.match(finished, /aria-valuenow="100"/);
      assert.match(finished, />100%/);
      assert.doesNotMatch(finished, /checkpoint-transfer-indeterminate|传输尚未开始/);
    }
  });

  await test('the latest real log remains visible and update time is described only as a state receipt', () => {
    const html = render({
      ...base,
      logs: 'first record\n2026-10-27T10:01:07Z 导出完成 <config>\n\n',
    });
    assert.match(html, /checkpoint-transfer-latest-log/);
    assert.match(html, /2026-10-27T10:01:07Z 导出完成 &lt;config&gt;/);
    assert.match(html, /任务创建于/);
    assert.match(html, /最近状态回执/);
    assert.match(html, /<details>[^]*first record/);
    assert.doesNotMatch(html, /最近真实输出/);
  });

  await test('elapsed time uses creation and finish timestamps rather than refreshed status timestamps', () => {
    const start = '2026-10-27T10:00:00Z';
    assert.equal(
      transferElapsed(start, '2026-10-27T10:01:24Z', Date.parse('2030-01-01T00:00:00Z')),
      '1分24秒',
    );
    assert.equal(transferElapsed(start, null, Date.parse('2026-10-27T11:01:24Z')), '1小时1分24秒');
    assert.equal(transferElapsed(start, null, Date.parse(start)), '0秒');
    assert.equal(transferElapsed('invalid', null, 0), undefined);
    assert.equal(transferElapsed(start, 'invalid'), undefined);
    assert.equal(transferElapsed(start, '2026-10-27T09:59:59Z'), undefined);
  });
} finally {
  await server.close();
}
