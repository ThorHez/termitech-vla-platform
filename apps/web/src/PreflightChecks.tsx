import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import type { PreflightCheck } from './types';

const labels: Record<string, string> = {
  runtime: '训练运行环境',
  local_paths: '本机资产路径',
  code_version: '实际执行代码版本',
  adapter: '训练模板与适配器',
  recipe_configuration: '配方配置解析',
  data_contract: '数据集与数据协议',
  dataset_fps: '数据集采样频率',
  norm_contract: '归一化参数',
  model_checkpoint: '模型检查点',
  training_entrypoint: '训练入口',
  training_cache: '训练缓存与数据加载权限',
  output_storage: '输出目录与磁盘空间',
  gpu_capacity: 'GPU 数量与配置',
  validation: '预检验证',
};

export function PreflightChecks({ checks }: { checks: readonly PreflightCheck[] }) {
  return (
    <div className="preflight-checks">
      {checks.map((check, index) => {
        const warning = check.ok === true && check.severity === 'warning';
        const state = !check.ok ? 'bad' : warning ? 'warning' : 'ok';
        const label = labels[check.name] || check.name;
        const status = !check.ok ? '未通过' : warning ? '警告' : '通过';
        return (
          <div
            className={state}
            key={`${check.name}-${index}`}
            role="group"
            aria-label={`${status}：${label}`}
          >
            {!check.ok ? (
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
            <span>{check.message}</span>
          </div>
        );
      })}
    </div>
  );
}
