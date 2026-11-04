import type { Gpu, Node, RunRequest } from './types';
import {
  sharedTrainingGpuError,
  supportsSharedTraining,
  validTrainingGpuPolicy,
} from './trainingGpuPolicy';

export function validRequestedGpuList(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.length <= 256 &&
    value.every(
      (uuid) =>
        typeof uuid === 'string' && !!uuid && uuid.length <= 200 && !/[\s\u0000-\u001f]/.test(uuid),
    ) &&
    new Set(value).size === value.length
  );
}

export function requestedGpuError(
  request: Pick<RunRequest, 'requested_gpu_uuids' | 'gpu_count' | 'training_gpu_policy'>,
  node?: Pick<Node, 'gpus'> & Partial<Pick<Node, 'metrics'>>,
  minimum = 1,
  userId?: string,
): string | undefined {
  const selected = request.requested_gpu_uuids;
  if (request.training_gpu_policy) {
    if (!validTrainingGpuPolicy(request.training_gpu_policy)) return '请填写有效的每卡显存预算。';
    if (!supportsSharedTraining(node)) return '当前 Agent 尚不支持共享显存训练，请先更新 Agent。';
    if (!selected?.length) return '共享显存模式需要明确勾选要使用的 GPU。';
  }
  if (selected === undefined) return undefined;
  if (!validRequestedGpuList(selected)) return '指定 GPU 列表无效，请重新选择。';
  if (!selected.length) return '请至少选择一张具体 GPU，或切换为自动分配。';
  if (selected.length < minimum) return `当前训练方式至少需要 ${minimum} 张 GPU。`;
  if (request.gpu_count !== selected.length) return '申请 GPU 数量与所选 GPU 不一致，请重新选择。';
  const available = new Set(node?.gpus?.map((gpu) => gpu.uuid));
  if (selected.some((uuid) => !available.has(uuid)))
    return '部分指定 GPU 已不属于当前机器或尚未上报，请重新选择；不会自动替换。';
  if (userId) {
    for (const gpu of node?.gpus || []) {
      if (!selected.includes(gpu.uuid)) continue;
      const reservationError = gpuReservationError(gpu, userId);
      if (reservationError) return reservationError;
    }
  }
  if (request.training_gpu_policy) {
    for (const gpu of node?.gpus || []) {
      if (!selected.includes(gpu.uuid)) continue;
      const issue = sharedTrainingGpuError(gpu, request.training_gpu_policy);
      if (issue) return issue;
    }
  }
  return undefined;
}

export function gpuReservationError(gpu: Gpu, userId?: string): string | undefined {
  const reservation = gpu.user_reservation;
  if (reservation && reservation.user_id !== userId)
    return `GPU 已由 ${reservation.display_name || reservation.username} 预留，请选择其他 GPU。`;
  return undefined;
}

export function gpuSelectionState(
  gpu: Gpu,
  userId?: string,
  policy?: RunRequest['training_gpu_policy'],
): 'available' | 'busy' | 'unknown' | 'reserved' | 'shareable' | 'insufficient' {
  if (gpuReservationError(gpu, userId)) return 'reserved';
  if (policy) {
    const issue = sharedTrainingGpuError(gpu, policy);
    if (!issue) return 'shareable';
    if (gpu.reserved_by || gpu.processes?.some((process) => process.ownership === 'platform'))
      return 'busy';
    return issue.startsWith('剩余') ? 'insufficient' : 'unknown';
  }
  if (gpu.reserved_by || (Array.isArray(gpu.processes) && gpu.processes.length > 0)) return 'busy';
  if (
    !Array.isArray(gpu.processes) ||
    typeof gpu.utilization_pct !== 'number' ||
    !Number.isFinite(gpu.utilization_pct) ||
    gpu.utilization_pct < 0 ||
    gpu.utilization_pct > 100 ||
    typeof gpu.memory_used_mb !== 'number' ||
    !Number.isFinite(gpu.memory_used_mb) ||
    gpu.memory_used_mb < 0 ||
    typeof gpu.memory_total_mb !== 'number' ||
    !Number.isFinite(gpu.memory_total_mb) ||
    gpu.memory_total_mb <= 0
  )
    return 'unknown';
  return gpu.utilization_pct <= 5 && gpu.memory_used_mb <= Math.max(128, gpu.memory_total_mb * 0.02)
    ? 'available'
    : 'busy';
}

export function requestedGpuSummary(
  selected: readonly string[] | undefined,
  node?: Pick<Node, 'gpus'>,
): string {
  if (selected === undefined) return '自动分配';
  if (!selected.length) return '尚未选择具体 GPU';
  return selected
    .map((uuid) => {
      const index = node?.gpus?.findIndex((gpu) => gpu.uuid === uuid) ?? -1;
      return index < 0 ? `未上报 GPU（${uuid}）` : `GPU ${index}`;
    })
    .join('、');
}
