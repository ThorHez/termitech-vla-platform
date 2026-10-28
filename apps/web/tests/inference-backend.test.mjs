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
  const { inferenceBackendError, resolvedInferenceBackendLabel, deploymentStartBlockedReason } =
    await server.ssrLoadModule('/src/policyDeployment.ts');
  const { DeploymentBackendField, DeploymentDetail } = await server.ssrLoadModule(
    '/src/PolicyDeployments.tsx',
  );
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const capable = {
    id: 'target',
    status: 'ONLINE',
    metrics: { capabilities: ['POLICY_DEPLOY_BACKEND_V1'] },
  };
  const legacy = { ...capable, metrics: { capabilities: ['POLICY_DEPLOY_START'] } };
  const field = (value, node = capable) =>
    renderToStaticMarkup(
      React.createElement(DeploymentBackendField, { value, node, onChange() {} }),
    );
  const deployment = {
    id: 'service',
    name: '后端明确的部署',
    node_id: 'target',
    profile_id: 'openpi',
    checkpoint_path: '/data/converted',
    config_name: 'pi05',
    port: 8000,
    state: 'STOPPED',
    checks: [],
    logs: '',
    created_at: '2026-10-28T00:00:00Z',
    updated_at: '2026-10-28T00:00:00Z',
  };
  const detail = (item) =>
    renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: () => true } },
        React.createElement(
          DataProvider,
          { enabled: false },
          React.createElement(DeploymentDetail, { id: item.id, initialItem: item, changed() {} }),
        ),
      ),
    );

  await test('backend choice follows checkpoint format and stays independent of GPU model', () => {
    for (const name of ['NVIDIA Thor', 'NVIDIA GeForce RTX 4090']) {
      const node = { ...capable, gpus: [{ uuid: 'gpu', name }] };
      assert.equal(inferenceBackendError('jax', node), undefined);
      assert.equal(inferenceBackendError('pytorch', node), undefined);
      const html = field('pytorch', node);
      assert.match(html, /value="pytorch" selected=""/);
      assert.match(html, /value="jax"/);
      assert.match(html, /value="auto"/);
      assert.match(html, /params\/ 使用 JAX/);
      assert.match(html, /model.safetensors 使用 PyTorch/);
      assert.match(html, /不会转换权重/);
    }
  });

  await test('legacy Agents keep automatic detection and explain explicit backend upgrade requirements', () => {
    assert.equal(inferenceBackendError('auto', undefined), undefined);
    assert.equal(inferenceBackendError('auto', legacy), undefined);
    assert.doesNotMatch(field('auto', legacy), /role="alert"/);
    for (const node of [
      undefined,
      legacy,
      { metrics: { capabilities: 'POLICY_DEPLOY_BACKEND_V1' } },
    ]) {
      for (const backend of ['jax', 'pytorch']) {
        assert.match(inferenceBackendError(backend, node), /更新 Agent/);
        assert.match(inferenceBackendError(backend, node), /权重同步可继续进行/);
      }
    }
    assert.match(field('jax', legacy), /role="alert"/);
  });

  await test('restart preserves the saved backend contract and requires support only for explicit frameworks', () => {
    assert.equal(deploymentStartBlockedReason(deployment, legacy, true), undefined);
    assert.equal(
      deploymentStartBlockedReason({ ...deployment, inference_backend: 'auto' }, legacy, true),
      undefined,
    );
    for (const inference_backend of ['jax', 'pytorch']) {
      const saved = { ...deployment, inference_backend };
      assert.match(deploymentStartBlockedReason(saved, legacy, true), /更新 Agent/);
      assert.equal(deploymentStartBlockedReason(saved, capable, true), undefined);
    }
  });

  await test('deployment details distinguish selected framework from Agent-confirmed framework', () => {
    const detected = detail({
      ...deployment,
      inference_backend: 'auto',
      resolved_backend: 'pytorch',
    });
    assert.match(detected, /<dt>推理框架<\/dt><dd>自动识别<\/dd>/);
    assert.match(detected, /<dt>实际推理框架<\/dt><dd>PyTorch<\/dd>/);
    const waiting = detail({ ...deployment, inference_backend: 'jax', resolved_backend: null });
    assert.match(waiting, /<dt>推理框架<\/dt><dd>JAX<\/dd>/);
    assert.match(waiting, /尚未识别，等待 Agent 上报/);
    assert.equal(resolvedInferenceBackendLabel('jax'), 'JAX');
    assert.equal(resolvedInferenceBackendLabel(undefined), '尚未识别，等待 Agent 上报');
  });
} finally {
  await server.close();
}
