import { Field } from './components';
import { useAuth } from './auth';
import { gpuSelectionState, requestedGpuError } from './trainingGpuSelection';
import type { Node, RunRequest } from './types';

const stateLabels = {
  available: '可用',
  busy: '占用中，将等待释放',
  unknown: '状态未知，将等待确认',
  reserved: '其他用户已预留',
};

export function TrainingGpuSelector({
  request,
  node,
  minimum,
  onChange,
}: {
  request: Pick<RunRequest, 'gpu_count' | 'requested_gpu_uuids'>;
  node?: Node;
  minimum: number;
  onChange: (selection: Pick<RunRequest, 'gpu_count' | 'requested_gpu_uuids'>) => void;
}) {
  const { user } = useAuth();
  const manual = request.requested_gpu_uuids !== undefined;
  const selected = request.requested_gpu_uuids || [];
  const error = requestedGpuError(request, node, minimum, user?.id);
  return (
    <div className="training-gpu-selection">
      <Field label="GPU 分配方式">
        <select
          aria-label="GPU 分配方式"
          value={manual ? 'specific' : 'automatic'}
          onChange={(event) => {
            onChange(
              event.target.value === 'specific'
                ? { requested_gpu_uuids: [], gpu_count: 0 }
                : {
                    requested_gpu_uuids: undefined,
                    gpu_count: Math.max(minimum, request.gpu_count),
                  },
            );
          }}
        >
          <option value="automatic">自动分配</option>
          <option value="specific" disabled={!node}>
            指定 GPU
          </option>
        </select>
      </Field>
      <Field
        label="申请 GPU 数量"
        hint={
          manual
            ? '数量跟随所选 GPU；占用中或状态未知时排队，不会换卡。'
            : minimum === 0
              ? '测试模板支持 0 GPU。'
              : '优先使用你的预留卡，再使用共享卡；资源不足时排队。'
        }
      >
        <input
          aria-label="申请 GPU 数量"
          type="number"
          required
          min={minimum}
          max={Math.max(minimum, node?.gpus?.length || 0)}
          step={1}
          value={request.gpu_count}
          readOnly={manual}
          onChange={(event) =>
            onChange({ requested_gpu_uuids: undefined, gpu_count: Number(event.target.value) })
          }
        />
      </Field>
      {manual && (
        <div className="training-gpu-options" aria-label="选择具体 GPU">
          {!node?.gpus?.length && <p className="field-hint">当前机器尚未上报 GPU，请等待刷新。</p>}
          {node?.gpus?.map((gpu, index) => {
            const state = gpuSelectionState(gpu, user?.id);
            return (
              <label
                key={gpu.uuid}
                className={`training-gpu-option ${selected.includes(gpu.uuid) ? 'selected' : ''}`}
              >
                <input
                  type="checkbox"
                  aria-label={`选择 GPU ${index}`}
                  checked={selected.includes(gpu.uuid)}
                  disabled={state === 'reserved' && !selected.includes(gpu.uuid)}
                  onChange={(event) => {
                    const next = event.target.checked
                      ? [...selected, gpu.uuid]
                      : selected.filter((uuid) => uuid !== gpu.uuid);
                    onChange({ requested_gpu_uuids: next, gpu_count: next.length });
                  }}
                />
                <span className="training-gpu-description">
                  <strong>
                    GPU {index} · {gpu.name}
                  </strong>
                  <span>
                    {typeof gpu.memory_total_mb === 'number'
                      ? `${(gpu.memory_total_mb / 1024).toFixed(0)} GB`
                      : '显存未知'}{' '}
                    · <span className={`training-gpu-state ${state}`}>{stateLabels[state]}</span>
                  </span>
                  {gpu.user_reservation && (
                    <span>
                      {gpu.user_reservation.user_id === user?.id
                        ? '为你预留'
                        : `${gpu.user_reservation.display_name || gpu.user_reservation.username} 已预留`}
                    </span>
                  )}
                  <code title={gpu.uuid}>
                    {gpu.uuid.length > 24
                      ? `${gpu.uuid.slice(0, 15)}…${gpu.uuid.slice(-6)}`
                      : gpu.uuid}
                  </code>
                </span>
              </label>
            );
          })}
          {error && (
            <p className="training-parameter-error" role="alert">
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
