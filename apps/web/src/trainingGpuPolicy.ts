import type { Gpu, Node, TrainingGpuPolicy } from './types';

export const TRAINING_SHARED_GPU_CAPABILITY = 'TRAINING_SHARED_GPU_V1';
export const TRAINING_GPU_RESERVE_MB = 1024;

export function validTrainingGpuPolicy(
  value: unknown,
  incomplete = false,
): value is TrainingGpuPolicy {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const policy = value as Record<string, unknown>;
  return (
    Object.keys(policy).length === 2 &&
    Object.keys(policy).every((key) => key === 'mode' || key === 'memory_mb') &&
    policy.mode === 'shared' &&
    typeof policy.memory_mb === 'number' &&
    (incomplete ||
      (Number.isSafeInteger(policy.memory_mb) &&
        policy.memory_mb >= 256 &&
        policy.memory_mb <= 1024 * 1024))
  );
}

export function supportsSharedTraining(node?: Pick<Node, 'metrics'>) {
  const capabilities = node?.metrics?.capabilities;
  return Array.isArray(capabilities) && capabilities.includes(TRAINING_SHARED_GPU_CAPABILITY);
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export function sharedTrainingGpuError(gpu: Gpu, policy: TrainingGpuPolicy): string | undefined {
  if (!validTrainingGpuPolicy(policy)) return '请填写有效的每卡显存预算。';
  if (gpu.reserved_by) return '该 GPU 正在被平台任务使用，请等待任务结束。';
  if (
    !finite(gpu.memory_total_mb) ||
    gpu.memory_total_mb <= 0 ||
    !finite(gpu.memory_used_mb) ||
    gpu.memory_used_mb < 0 ||
    gpu.memory_used_mb > gpu.memory_total_mb ||
    !finite(gpu.memory_free_mb) ||
    gpu.memory_free_mb < 0 ||
    gpu.memory_free_mb + gpu.memory_used_mb > gpu.memory_total_mb + 1 ||
    !finite(gpu.utilization_pct) ||
    gpu.utilization_pct < 0 ||
    gpu.utilization_pct > 100 ||
    !Array.isArray(gpu.processes)
  )
    return 'GPU 遥测不完整，请等待 Agent 更新状态。';
  for (const process of gpu.processes) {
    if (process.ownership === 'platform') return '该 GPU 正在被平台任务使用，请等待任务结束。';
    if (
      !Number.isSafeInteger(process.pid) ||
      process.pid <= 0 ||
      !['compute', 'graphics', 'compute_graphics'].includes(process.kind || '') ||
      ![undefined, null, 'external', 'platform'].includes(process.ownership)
    )
      return 'GPU 进程类型未知，暂不能共享使用。';
  }
  if (gpu.memory_free_mb < policy.memory_mb + TRAINING_GPU_RESERVE_MB)
    return `剩余 ${(gpu.memory_free_mb / 1024).toFixed(1)} GiB，不足以容纳每卡预算 ${(policy.memory_mb / 1024).toFixed(2)} GiB 和 1 GiB 余量。`;
  return undefined;
}

export function defaultTrainingMemoryBudget(node?: Pick<Node, 'gpus'>) {
  const free = (node?.gpus || []).map((gpu) => gpu.memory_free_mb).filter(finite);
  const gib = free.length
    ? Math.max(0.25, Math.min(50, Math.floor(Math.max(...free) / 1024) - 2))
    : 32;
  return gib * 1024;
}

export function trainingGpuPolicySummary(policy?: TrainingGpuPolicy | null) {
  return policy?.mode === 'shared'
    ? `共享显存 · 每卡预算 ${(policy.memory_mb / 1024).toFixed(2)} GiB`
    : '独占使用';
}
