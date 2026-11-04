import { Field } from './components';
import { useAuth } from './auth';
import { gpuSelectionState, requestedGpuError } from './trainingGpuSelection';
import type { Node, RunRequest } from './types';
import { defaultTrainingMemoryBudget, supportsSharedTraining } from './trainingGpuPolicy';

const stateLabels = {
  available: '可用',
  busy: '占用中，将等待释放',
  unknown: '状态未知，将等待确认',
  reserved: '其他用户已预留',
  shareable: '可共享使用',
  insufficient: '剩余显存不足',
};

export function TrainingGpuSelector({
  request,
  node,
  minimum,
  onChange,
  allowShared = true,
}: {
  request: Pick<RunRequest, 'gpu_count' | 'requested_gpu_uuids' | 'training_gpu_policy'>;
  node?: Node;
  minimum: number;
  onChange: (
    selection: Pick<RunRequest, 'gpu_count' | 'requested_gpu_uuids' | 'training_gpu_policy'>,
  ) => void;
  allowShared?: boolean;
}) {
  const { user } = useAuth();
  const manual = request.requested_gpu_uuids !== undefined;
  const selected = request.requested_gpu_uuids || [];
  const shared = request.training_gpu_policy?.mode === 'shared';
  const error = requestedGpuError(request, node, minimum, user?.id);
  return (
    <div className="training-gpu-selection">
      {allowShared && (
        <Field
          label="GPU 使用模式"
          hint="独占要求整卡空闲；共享可使用已有外部进程的显卡，需要指定显卡并设置显存预算。"
        >
          <select
            aria-label="GPU 使用模式"
            value={shared ? 'shared' : 'exclusive'}
            onChange={(event) =>
              onChange(
                event.target.value === 'shared'
                  ? {
                      gpu_count: selected.length,
                      requested_gpu_uuids: selected,
                      training_gpu_policy: {
                        mode: 'shared',
                        memory_mb: defaultTrainingMemoryBudget(node),
                      },
                    }
                  : {
                      gpu_count: request.gpu_count,
                      requested_gpu_uuids: request.requested_gpu_uuids,
                      training_gpu_policy: undefined,
                    },
              )
            }
          >
            <option value="exclusive">独占使用（默认）</option>
            <option value="shared" disabled={!supportsSharedTraining(node)}>
              共享剩余显存
            </option>
          </select>
          {!supportsSharedTraining(node) && (
            <span className="field-hint">当前 Agent 不支持共享显存训练，更新后可使用。</span>
          )}
        </Field>
      )}
      {shared && (
        <Field
          label="每卡显存预算（GiB）"
          required
          hint="每张卡需额外保留 1 GiB 余量。预算控制 JAX 显存池；已有进程继续增长或训练所需显存超过预算时仍可能显存不足。"
        >
          <input
            aria-label="每卡显存预算（GiB）"
            type="number"
            min={0.25}
            max={1024}
            step={0.25}
            required
            value={
              Number.isFinite(request.training_gpu_policy?.memory_mb)
                ? (request.training_gpu_policy?.memory_mb || 0) / 1024
                : ''
            }
            onChange={(event) =>
              onChange({
                gpu_count: request.gpu_count,
                requested_gpu_uuids: request.requested_gpu_uuids,
                training_gpu_policy: {
                  mode: 'shared',
                  memory_mb: event.target.value === '' ? NaN : Number(event.target.value) * 1024,
                },
              })
            }
          />
        </Field>
      )}
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
          <option value="automatic" disabled={shared}>
            自动分配
          </option>
          <option value="specific" disabled={!node}>
            指定 GPU
          </option>
        </select>
      </Field>
      <Field
        label="申请 GPU 数量"
        hint={
          shared
            ? '共享模式只使用勾选的显卡，按每卡剩余显存检查；平台任务占用和其他用户预约仍会拦截。'
            : manual
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
            const state = gpuSelectionState(gpu, user?.id, request.training_gpu_policy);
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
                  {shared && (
                    <span>
                      当前剩余{' '}
                      {typeof gpu.memory_free_mb === 'number'
                        ? `${(gpu.memory_free_mb / 1024).toFixed(1)} GiB`
                        : '未知'}
                    </span>
                  )}
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
