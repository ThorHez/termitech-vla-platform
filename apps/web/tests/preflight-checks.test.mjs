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
  const { preflightValidationAsset } = await server.ssrLoadModule('/src/datasetValidationHelpers.ts');
  const render = (checks, summarizedSharedNormError, datasetValidationNotice) => renderToStaticMarkup(
    React.createElement(PreflightChecks, { checks, summarizedSharedNormError, datasetValidationNotice }),
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

  const baseValidationCode = 'DATASET_BASE_VALIDATION_REQUIRED';
  const validationMessage = `${baseValidationCode}: 数据集 /data/项目 A/demo 的基础校验证据缺失或已失效，请先在数据集配置中执行基础校验`;
  const validationAsset = { id: 'target', kind: 'DATASET', node_id: 'node-1', path: '/data//项目 A/./demo/' };
  const failedValidationPreview = {
    status: 'FAILED',
    request: { node_id: 'node-1', profile_id: 'runtime-2', inputs: { dataset_0: 'other', dataset_1: 'target', norm_1: 'norm' } },
    error: { code: baseValidationCode, message: validationMessage },
  };

  await test('missing receipts resolve the matching frozen dataset for current and legacy error wording', () => {
    const assets = [{ ...validationAsset, id: 'other', path: '/data/项目 A/demo-longer' }, validationAsset];
    for (const reason of [
      '的基础校验证据缺失或已失效，请先在数据集配置中执行基础校验',
      '缺少匹配的基础校验结果，或数据、登记配置、校验环境已变化',
      '的 ReCap 监督基础校验证据缺失或已失效，请先在数据集配置中执行基础校验',
    ]) {
      const preview = { ...failedValidationPreview, error: { code: baseValidationCode, message: `数据集 /data/项目 A/demo ${reason}` } };
      assert.equal(preflightValidationAsset(preview, assets), validationAsset);
    }
    assert.equal(preflightValidationAsset(failedValidationPreview, assets), validationAsset);
  });

  await test('refresh does not target unrelated assets, other machines, or ambiguous paths', () => {
    const otherAssets = [
      { ...validationAsset, node_id: 'another-node' },
      { ...validationAsset, id: 'other', path: '/data/项目 A/demo-longer' },
      { ...validationAsset, id: 'unselected' },
      { ...validationAsset, id: 'norm' },
    ];
    assert.equal(preflightValidationAsset(failedValidationPreview, otherAssets), undefined);
    assert.equal(preflightValidationAsset(failedValidationPreview, [validationAsset, { ...validationAsset, id: 'other' }]), undefined);
    assert.equal(preflightValidationAsset(failedValidationPreview, [{ ...validationAsset, path: '/data/项目 A/demo/../demo' }]), undefined);
    assert.equal(preflightValidationAsset({ ...failedValidationPreview, status: 'READY' }, [validationAsset]), undefined);
  });

  await test('missing-receipt checks point to one action while unrelated failures keep their detail', () => {
    const html = render([
      { name: 'dataset_timestamps', ok: false, message: validationMessage },
      { name: 'validation', ok: false, message: validationMessage },
      { name: 'norm_contract', ok: false, message: '归一化参数维度不匹配' },
    ], undefined, 'required');
    assert.match(html, /请处理上方的数据集基础校验提示/);
    assert.match(html, /归一化参数维度不匹配/);
    assert.doesNotMatch(html, /DATASET_BASE_VALIDATION_REQUIRED|\/data\/项目 A/);
  });

  await test('refreshed evidence leaves the affected checks awaiting re-preflight and preserves other failures', () => {
    const html = render([
      { name: 'dataset_timestamps', ok: false, message: validationMessage },
      { name: 'validation', ok: false, code: baseValidationCode },
      { name: 'norm_contract', ok: false, message: '归一化参数维度不匹配' },
    ], undefined, 'refreshed');
    assert.match(html, /class="pending" role="group" aria-label="待重新预检：真实数据时间戳与采样率"/);
    assert.match(html, /class="pending" role="group" aria-label="待重新预检：预检验证"/);
    assert.match(html, /class="bad" role="group" aria-label="未通过：归一化参数"/);
    assert.doesNotMatch(html, /请处理上方的数据集基础校验提示|DATASET_BASE_VALIDATION_REQUIRED/);
  });
} finally {
  await server.close();
}
