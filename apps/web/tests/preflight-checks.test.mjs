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
  const { PreflightChecks } = await server.ssrLoadModule('/src/PreflightChecks.tsx');
  const render = (checks) => renderToStaticMarkup(React.createElement(PreflightChecks, { checks }));
  const fps = { name: 'dataset_fps', ok: true, severity: 'warning', code: 'DATASET_FPS_MISMATCH', message: '登记 20 Hz，元信息 60 Hz；使用实际采样率。' };

  await test('a successful FPS warning has a warning row, triangle and accessible Chinese status', () => {
    const html = render([fps]);
    assert.match(html, /class="warning" role="group" aria-label="警告：数据集采样频率"/);
    assert.match(html, /lucide-triangle-alert/);
    assert.match(html, /aria-hidden="true"/);
    assert.match(html, /class="preflight-warning-label">警告<\/small>/);
    assert.match(html, /登记 20 Hz，元信息 60 Hz；使用实际采样率。/);
    assert.doesNotMatch(html, /class="ok"|class="bad"|未通过/);
  });

  await test('a failed check remains a hard error even when its severity claims warning', () => {
    const html = render([{ ...fps, ok: false }]);
    assert.match(html, /class="bad" role="group" aria-label="未通过：数据集采样频率"/);
    assert.match(html, /lucide-circle-x/);
    assert.doesNotMatch(html, /class="warning"|class="ok"|preflight-warning-label|lucide-triangle-alert/);
  });

  await test('warnings and actual failures coexist without recoloring or hiding either result', () => {
    const html = render([fps, { name: 'norm_contract', ok: false, message: '归一化参数维度不匹配' },
      { name: 'data_contract', ok: true, message: '数据协议通过' }]);
    assert.equal((html.match(/class="warning"/g) || []).length, 1);
    assert.equal((html.match(/class="bad"/g) || []).length, 1);
    assert.equal((html.match(/class="ok"/g) || []).length, 1);
    assert.match(html, /未通过：归一化参数/);
    assert.match(html, /归一化参数维度不匹配/);
    assert.match(html, /通过：数据集与数据协议/);
  });

  await test('legacy checks remain compatible and unknown names and messages are escaped', () => {
    const html = render([{ name: 'runtime', ok: true, message: 'Python 可用' },
      { name: '<custom>', ok: false, message: '<script>unsafe</script>' }, { name: 'gpu_capacity', ok: true }]);
    assert.match(html, /通过：训练运行环境/);
    assert.match(html, /Python 可用/);
    assert.match(html, /&lt;custom&gt;/);
    assert.match(html, /&lt;script&gt;unsafe&lt;\/script&gt;/);
    assert.doesNotMatch(html, /class="warning"|<script>/);
  });
} finally {
  await server.close();
}
