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
  const { PreflightChecks, sharedNormIncompatibilityDetail } = await server.ssrLoadModule('/src/PreflightChecks.tsx');
  const render = (checks, summarizedSharedNormError) => renderToStaticMarkup(
    React.createElement(PreflightChecks, { checks, summarizedSharedNormError }),
  );
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

  await test('real timestamp mismatches are red blockers while metadata FPS differences remain warnings', () => {
    const html = render([fps, { name: 'dataset_timestamps', ok: false,
      message: '生效 20 Hz，期望间隔 0.05 秒，实际间隔 0.016667 秒；请核对采样率或先重采样。' }]);
    assert.match(html, /class="bad" role="group" aria-label="未通过：真实数据时间戳与采样率"/);
    assert.match(html, /期望间隔 0.05 秒/);
    assert.match(html, /实际间隔 0.016667 秒/);
    assert.equal((html.match(/class="warning"/g) || []).length, 1);
    assert.equal((html.match(/class="bad"/g) || []).length, 1);
  });

  await test('statistics entrypoints have an accessible Chinese label', () => {
    const html = render([{ name: 'norm_entrypoint', ok: true, message: '统计脚本可用' }]);
    assert.match(html, /aria-label="通过：统计脚本入口"/);
    assert.match(html, /<strong>统计脚本入口<\/strong>/);
    assert.doesNotMatch(html, /norm_entrypoint/);
  });

  await test('legacy shared norm incompatibilities explain the check in Chinese', () => {
    const message = 'SHARED_NORM_INVALID: source state/action layouts, ordering, encoding, units or numeric processing differ';
    const html = render([{ name: 'validation', ok: false, message }]);
    assert.match(html, /未通过：数据集能否共用归一化参数/);
    assert.match(html, /各组绑定的 NORM 仅作参考/);
    assert.doesNotMatch(html, /SHARED_NORM_INVALID|source state\/action/);
  });

  await test('the failed check points to the shared norm notice when that notice shows the details', () => {
    const detail = '本次多数据集训练会自动计算共享归一化参数。第 4 组「collect_1029_human」的动作表示为关节。';
    const message = 'SHARED_NORM_INVALID: ' + detail;
    assert.equal(sharedNormIncompatibilityDetail(message), detail);
    const html = render([{ name: 'validation', ok: false, message }], message);
    assert.match(html, /未通过：数据集能否共用归一化参数/);
    assert.match(html, /详见上方的失败原因/);
    assert.doesNotMatch(html, /collect_1029_human|SHARED_NORM_INVALID/);
    assert.equal(sharedNormIncompatibilityDetail('SHARED_NORM_INVALID: 共享统计文件在预检期间改变'), undefined);
  });
} finally {
  await server.close();
}
