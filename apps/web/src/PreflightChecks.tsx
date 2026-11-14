import { AlertTriangle, CheckCircle2, Circle, XCircle } from 'lucide-react';
import { isPreflightDatasetValidationCheck } from './datasetValidationHelpers';
import type { PreflightCheck } from './types';

const labels: Record<string, string> = {
  runtime: '训练运行环境',
  local_paths: '本机资产路径',
  code_version: '实际执行代码版本',
  adapter: '训练模板与适配器',
  recipe_configuration: '配方配置解析',
  data_contract: '数据集与数据协议',
  dataset_fps: '数据集采样频率',
  dataset_timestamps: '真实数据时间戳与采样率',
  dataset_samples: '基础校验证据与数据处理配置',
  recap_stage: 'ReCap 阶段校验',
  norm_contract: '归一化参数',
  model_checkpoint: '模型检查点',
  resume_checkpoint: '续训检查点',
  training_entrypoint: '训练入口',
  norm_entrypoint: '统计脚本入口',
  training_cache: '训练缓存与数据加载权限',
  output_storage: '输出目录与磁盘空间',
  output_isolation: '输出目录与代码、资产隔离',
  gpu_capacity: 'GPU 数量与配置',
  validation: '预检验证',
};

const sharedNormCodePrefix = 'SHARED_NORM_INVALID: ';
const legacySharedNormMismatch =
  'source state/action layouts, ordering, encoding, units or numeric processing differ';

export function sharedNormIncompatibilityDetail(message?: string) {
  if (!message?.startsWith(sharedNormCodePrefix)) return undefined;
  const detail = message.slice(sharedNormCodePrefix.length);
  if (detail === legacySharedNormMismatch) {
    return '各组数据的状态或动作定义不一致。自动计算共享归一化时，各组绑定的 NORM 仅作参考。请在「训练数据集」中核对各组的状态字段顺序、动作表示和单位。';
  }
  if (
    detail.startsWith('本次多数据集训练会自动计算共享归一化参数') ||
    detail.startsWith('本次训练使用共享归一化参数')
  ) {
    return detail;
  }
}

export function PreflightChecks({
  checks,
  summarizedSharedNormError,
  datasetValidationNotice,
}: {
  checks: readonly PreflightCheck[];
  summarizedSharedNormError?: string;
  datasetValidationNotice?: 'required' | 'refreshed';
}) {
  return (
    <div className="preflight-checks">
      {checks.map((check, index) => {
        const warning = check.ok === true && check.severity === 'warning';
        const sharedNormDetail = check.ok
          ? undefined
          : sharedNormIncompatibilityDetail(check.message);
        const label = sharedNormDetail
          ? '数据集能否共用归一化参数'
          : labels[check.name] || check.name;
        const summarizedDatasetValidation =
          datasetValidationNotice && isPreflightDatasetValidationCheck(check);
        const detail = sharedNormDetail
          ? check.message === summarizedSharedNormError
            ? '详见上方的失败原因。'
            : sharedNormDetail
          : check.message;
        const pending = summarizedDatasetValidation && datasetValidationNotice === 'refreshed';
        const state = pending ? 'pending' : !check.ok ? 'bad' : warning ? 'warning' : 'ok';
        const message = pending
          ? '等待重新预检。'
          : summarizedDatasetValidation
            ? '请处理上方的数据集基础校验提示。'
            : detail;
        const status = pending ? '待重新预检' : !check.ok ? '未通过' : warning ? '警告' : '通过';
        return (
          <div
            className={state}
            key={`${check.name}-${index}`}
            role="group"
            aria-label={`${status}：${label}`}
          >
            {pending ? (
              <Circle size={17} aria-hidden="true" />
            ) : !check.ok ? (
              <XCircle size={17} aria-hidden="true" />
            ) : warning ? (
              <AlertTriangle size={17} aria-hidden="true" />
            ) : (
              <CheckCircle2 size={17} aria-hidden="true" />
            )}
            <strong>
              {label}
              {warning && <small className="preflight-warning-label">警告</small>}
            </strong>
            <span>{message}</span>
          </div>
        );
      })}
    </div>
  );
}
