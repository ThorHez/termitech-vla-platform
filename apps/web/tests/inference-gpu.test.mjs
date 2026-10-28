import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});
try {
  const { inferenceGpuError, inferenceMemoryMb, sharedInferenceSupported, inferenceGpuMemory } =
    await server.ssrLoadModule('/src/inferenceGpu.ts');
  const { gpuIdle } = await server.ssrLoadModule('/src/data.tsx');
  const node = {
    id: 'desktop-machine',
    name: '4090 desktop',
    status: 'ONLINE',
    metrics: { capabilities: ['POLICY_DEPLOY_SHARED_GPU_V1'] },
  };
  const graphics = {
    uuid: 'gpu-desktop',
    name: 'NVIDIA GeForce RTX 4090',
    memory_total_mb: 24564,
    memory_used_mb: 1460.44,
    memory_free_mb: 22500,
    utilization_pct: 18,
    processes: [{ pid: 2728, memory_mb: 209.84, kind: 'graphics', ownership: 'external' }],
  };
  await test('desktop graphics may share inference while training stays exclusively idle', () => {
    assert.equal(inferenceGpuError(graphics, 'shared', 16384, node), undefined);
    assert.equal(gpuIdle(graphics), false);
    assert.match(inferenceGpuError(graphics, 'exclusive', null, node), /独占模式/);
    assert.equal(
      inferenceGpuError({ ...graphics, utilization_pct: 100 }, 'shared', 16384, node),
      undefined,
    );
  });
  await test('GPU reservations and compute processes cannot be bypassed by shared mode', () => {
    assert.match(
      inferenceGpuError({ ...graphics, reserved_by: 'deployment-a' }, 'shared', 16384, node),
      /已被平台/,
    );
    for (const kind of ['compute', 'compute_graphics', 'unknown', undefined]) {
      const gpu = { ...graphics, processes: [{ pid: 4, kind }] };
      assert.match(
        inferenceGpuError(gpu, 'shared', 16384, node),
        /计算进程|类型未知|确认 GPU 进程类型/,
      );
    }
    assert.match(
      inferenceGpuError(
        { ...graphics, processes: [{ pid: 4, kind: 'graphics', ownership: 'platform' }] },
        'shared',
        16384,
        node,
      ),
      /计算进程或平台任务/,
    );
  });
  await test('shared memory admission uses reported free memory plus the full 1 GiB reserve', () => {
    const boundary = { ...graphics, memory_used_mb: 1000, memory_free_mb: 17408 };
    assert.equal(inferenceGpuError(boundary, 'shared', 16384, node), undefined);
    assert.match(
      inferenceGpuError({ ...boundary, memory_free_mb: 17407 }, 'shared', 16384, node),
      /需要预算.*1 GiB/,
    );
    for (const free of [undefined, null, NaN, Infinity, -1, 25000, '22500', true])
      assert.match(
        inferenceGpuError({ ...graphics, memory_free_mb: free }, 'shared', 16384, node),
        /可靠.*显存/,
      );
    assert.match(
      inferenceGpuError({ ...graphics, memory_used_mb: 24500 }, 'shared', 16384, node),
      /可靠.*显存/,
    );
  });
  await test('unknown process telemetry and invalid process identities fail closed', () => {
    for (const processes of [
      undefined,
      null,
      'graphics',
      [{ pid: 0, kind: 'graphics' }],
      [{ pid: true, kind: 'graphics' }],
    ])
      assert.ok(inferenceGpuError({ ...graphics, processes }, 'shared', 16384, node));
    assert.equal(
      inferenceGpuError({ ...graphics, processes: [] }, 'shared', 16384, node),
      undefined,
    );
  });
  await test('old Agents cannot authorize sharing and invalid budgets are explicit', () => {
    for (const old of [
      undefined,
      { metrics: {} },
      { metrics: { capabilities: ['POLICY_DEPLOY_START'] } },
    ]) {
      assert.equal(sharedInferenceSupported(old), false);
      assert.match(inferenceGpuError(graphics, 'shared', 16384, old), /更新 Agent/);
    }
    for (const budget of [null, 0, 255, 1048577, 1.5, NaN, Infinity, true])
      assert.match(inferenceGpuError(graphics, 'shared', budget, node), /预计推理显存/);
  });
  await test('GiB input preserves exact MiB budgets and rejects blanks and non-finite values', () => {
    assert.equal(inferenceMemoryMb('16'), 16384);
    assert.equal(inferenceMemoryMb('0.25'), 256);
    assert.equal(inferenceMemoryMb('1024'), 1048576);
    for (const value of ['', ' ', 'NaN', 'Infinity', '-1', '0.249', '1025', '16.0001'])
      assert.equal(inferenceMemoryMb(value), null, value);
    assert.equal(inferenceGpuMemory(17408), '17.00 GiB');
    assert.equal(inferenceGpuMemory(NaN), '未知');
  });
} finally {
  await server.close();
}
