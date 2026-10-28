import type { Gpu, Node } from './types';

export type InferenceGpuMode = 'exclusive' | 'shared';
export const inferenceGpuReserveMb = 1024;

export function sharedInferenceSupported(node?: Node) {
  return (
    Array.isArray(node?.metrics?.capabilities) &&
    node.metrics.capabilities.includes('POLICY_DEPLOY_SHARED_GPU_V1')
  );
}

export function inferenceMemoryMb(gib: string): number | null {
  if (!gib.trim()) return null;
  const value = Number(gib) * 1024;
  return Number.isSafeInteger(value) && value >= 256 && value <= 1048576 ? value : null;
}

const finiteMemory = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

export const inferenceGpuMemory = (memoryMb?: number | null) =>
  finiteMemory(memoryMb) ? `${(memoryMb / 1024).toFixed(2)} GiB` : '未知';

export function inferenceGpuError(
  gpu: Gpu | undefined,
  mode: InferenceGpuMode,
  memoryMb: number | null,
  node?: Node,
): string | undefined {
  if (mode === 'shared') {
    if (!sharedInferenceSupported(node)) return '当前 Agent 尚不支持共享 GPU，请先更新 Agent。';
    if (!Number.isSafeInteger(memoryMb) || memoryMb! < 256 || memoryMb! > 1048576)
      return '请填写 0.25–1024 GiB 的预计推理显存，精确到整数 MiB。';
  } else if (mode !== 'exclusive') {
    return 'GPU 使用方式无效。';
  } else if (memoryMb !== null) {
    return '独占模式无需填写共享显存预算。';
  }
  if (!gpu) return '请选择推理 GPU。';
  if (gpu.reserved_by) return 'GPU 已被平台训练或部署预约。';
  if (!Array.isArray(gpu.processes)) return 'GPU 进程状态未知，请等待可靠采样。';
  if (
    !finiteMemory(gpu.memory_total_mb) ||
    gpu.memory_total_mb <= 0 ||
    !finiteMemory(gpu.memory_used_mb) ||
    gpu.memory_used_mb > gpu.memory_total_mb
  )
    return 'GPU 显存信息缺失或无效。';
  if (mode === 'exclusive') {
    if (gpu.processes.length) return '独占模式要求 GPU 没有现有进程。';
    if (
      typeof gpu.utilization_pct !== 'number' ||
      !Number.isFinite(gpu.utilization_pct) ||
      gpu.utilization_pct < 0 ||
      gpu.utilization_pct > 100
    )
      return 'GPU 利用率信息缺失或无效。';
    if (gpu.utilization_pct > 5 || gpu.memory_used_mb > Math.max(128, gpu.memory_total_mb * 0.02))
      return '独占模式要求利用率不超过 5%，已用显存不超过总量的 2%（最低 128 MiB）。';
    return undefined;
  }
  if (
    !finiteMemory(gpu.memory_free_mb) ||
    gpu.memory_free_mb > gpu.memory_total_mb ||
    gpu.memory_free_mb + gpu.memory_used_mb > gpu.memory_total_mb + 1
  )
    return '缺少可靠的 GPU 剩余显存信息，请更新 Agent 或等待重新采样。';
  for (const process of gpu.processes) {
    if (
      !Number.isSafeInteger(process?.pid) ||
      process.pid <= 0 ||
      !['graphics', 'compute', 'compute_graphics', 'unknown'].includes(process.kind || '')
    )
      return '无法确认 GPU 进程类型，请更新 Agent 后重试。';
    if (process.kind === 'unknown') return 'GPU 存在类型未知的进程，暂不能共享部署。';
    if (process.kind !== 'graphics' || process.ownership === 'platform')
      return 'GPU 存在计算进程或平台任务，共享模式仅允许与图形桌面共存。';
  }
  const required = memoryMb! + inferenceGpuReserveMb;
  if (gpu.memory_free_mb < required)
    return `剩余 ${inferenceGpuMemory(gpu.memory_free_mb)}，需要预算 ${inferenceGpuMemory(memoryMb)} + 1 GiB 余量。`;
  return undefined;
}
