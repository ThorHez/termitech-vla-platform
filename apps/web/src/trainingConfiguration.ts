import type { Asset, AutoTrainingConfig, DatasetBinding, GeneratedTrainingConfig } from './types';

export const DEFAULT_AUTO_CONFIG: AutoTrainingConfig = {
  model_type: 'pi05',
  finetune_mode: 'full',
  action_horizon: 50,
  action_dim: 32,
};

export interface DatasetActionHorizonSource {
  bindingIndex: number;
  assetId: string;
  assetName: string;
  actionKey?: string;
  shape?: number[];
  sourceHorizon?: number;
  kind: 'single-step' | 'packed' | 'unknown';
}

export interface DatasetActionHorizonDiagnostic {
  code:
    | 'ASSET_UNAVAILABLE'
    | 'ACTION_FIELD_UNAVAILABLE'
    | 'ACTION_FIELD_AMBIGUOUS'
    | 'ACTION_SHAPE_MISSING'
    | 'ACTION_SHAPE_INVALID'
    | 'ACTION_SHAPE_UNSUPPORTED'
    | 'SOURCE_HORIZON_INCONSISTENT'
    | 'ACTION_HORIZON_CONFLICT'
    | 'ACTION_HORIZON_OUT_OF_RANGE';
  severity: 'warning' | 'error';
  message: string;
  bindingIndex?: number;
}

export interface DatasetActionHorizonInference {
  recommendedHorizon?: number;
  sources: DatasetActionHorizonSource[];
  diagnostics: DatasetActionHorizonDiagnostic[];
  errors: string[];
  warnings: string[];
}

const record = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;

/** Recommend only a stored [H,D] sequence length; raw [D] still needs future sampling.
 * This reads registered source facts and never changes the user's training config.
 */
export function inferDatasetActionHorizon(
  bindings: readonly DatasetBinding[],
  assets: readonly Asset[],
): DatasetActionHorizonInference {
  const sources: DatasetActionHorizonSource[] = [];
  const diagnostics: DatasetActionHorizonDiagnostic[] = [];
  const report = (
    source: DatasetActionHorizonSource,
    code: DatasetActionHorizonDiagnostic['code'],
    severity: DatasetActionHorizonDiagnostic['severity'],
    message: string,
  ) =>
    diagnostics.push({
      code,
      severity,
      message: `${source.assetName}：${message}`,
      bindingIndex: source.bindingIndex,
    });

  for (const [bindingIndex, binding] of bindings.entries()) {
    if (!binding.dataset_asset_id) continue;
    const asset = assets.find(
      (item) => item.id === binding.dataset_asset_id && item.kind === 'DATASET',
    );
    const source: DatasetActionHorizonSource = {
      bindingIndex,
      assetId: binding.dataset_asset_id,
      assetName: asset?.name || `数据集 ${bindingIndex + 1}`,
      kind: 'unknown',
    };
    sources.push(source);
    if (!asset) {
      report(
        source,
        'ASSET_UNAVAILABLE',
        'warning',
        '尚未取得所选数据集的字段信息，预测长度将在远端预检中核验。',
      );
      continue;
    }
    const profile = record(asset.metadata?.dataset_profile);
    const fields = Array.isArray(profile?.field_annotations)
      ? profile.field_annotations
          .map(record)
          .filter((field): field is Record<string, unknown> => !!field)
      : [];
    const composition =
      binding.processor?.name === 'ComposableDataConfig'
        ? record(binding.processor.parameters?.composition)
        : undefined;
    const selectedKey =
      typeof composition?.action_key === 'string' ? composition.action_key.trim() : '';
    const candidates = fields.filter(
      (field) =>
        field.role === 'action' &&
        typeof field.key === 'string' &&
        !!field.key.trim() &&
        (!selectedKey || field.key.trim() === selectedKey),
    );
    if (selectedKey) source.actionKey = selectedKey;
    if (candidates.length !== 1) {
      report(
        source,
        candidates.length ? 'ACTION_FIELD_AMBIGUOUS' : 'ACTION_FIELD_UNAVAILABLE',
        'warning',
        candidates.length
          ? '存在多个动作字段候选，请先在本次训练的字段组合中确定动作字段，当前不推断预测长度。'
          : selectedKey
            ? `所选动作字段 ${selectedKey} 没有唯一的动作标签，预测长度将在远端预检中核验。`
            : '尚无可用的动作字段标签，预测长度将在远端预检中核验。',
      );
      continue;
    }
    const field = candidates[0];
    source.actionKey = (field.key as string).trim();
    if (field.shape === undefined) {
      report(
        source,
        'ACTION_SHAPE_MISSING',
        'warning',
        '缺少原始动作形状；单独登记的 horizon 不能证明已存动作序列长度，请手动确认训练预测长度。',
      );
      continue;
    }
    if (
      !Array.isArray(field.shape) ||
      !field.shape.length ||
      field.shape.some((size) => !Number.isSafeInteger(size) || size < 1 || size > 2 ** 31 - 1)
    ) {
      report(
        source,
        'ACTION_SHAPE_INVALID',
        'error',
        `动作字段 ${source.actionKey} 的原始形状无效，请重新读取或修正数据集字段。`,
      );
      continue;
    }
    source.shape = [...field.shape];
    if (![1, 2].includes(source.shape.length)) {
      report(
        source,
        'ACTION_SHAPE_UNSUPPORTED',
        'warning',
        `动作形状 [${source.shape.join(', ')}] 无法确定时间轴，当前不推断预测长度。`,
      );
      continue;
    }
    source.kind = source.shape.length === 1 ? 'single-step' : 'packed';
    source.sourceHorizon = source.kind === 'single-step' ? 1 : source.shape[0];
    if (field.horizon !== undefined && field.horizon !== source.sourceHorizon) {
      report(
        source,
        'SOURCE_HORIZON_INCONSISTENT',
        'error',
        `动作字段 ${source.actionKey} 登记的 horizon 与原始形状 [${source.shape.join(', ')}] 不一致，请先修正字段标签。`,
      );
    }
    if (source.kind === 'packed' && source.sourceHorizon > 1000) {
      report(
        source,
        'ACTION_HORIZON_OUT_OF_RANGE',
        'error',
        `已存动作序列长度 ${source.sourceHorizon} 超出训练预测长度支持的 1–1000 范围。`,
      );
    }
  }

  const packed = sources.filter((source) => source.kind === 'packed');
  const horizons = [...new Set(packed.map((source) => source.sourceHorizon!))];
  if (horizons.length > 1) {
    diagnostics.push({
      code: 'ACTION_HORIZON_CONFLICT',
      severity: 'error',
      message: `所选数据集的已存动作序列长度不一致（${packed.map((source) => `${source.assetName}：${source.sourceHorizon}`).join('；')}），不能直接混合训练，请调整数据集或处理方式。`,
    });
  }
  const errors = diagnostics
    .filter((item) => item.severity === 'error')
    .map((item) => item.message);
  const warnings = diagnostics
    .filter((item) => item.severity === 'warning')
    .map((item) => item.message);
  return {
    ...(horizons.length === 1 && !errors.length ? { recommendedHorizon: horizons[0] } : {}),
    sources,
    diagnostics,
    errors,
    warnings,
  };
}

export function generatedConfigAvailable(
  config?: GeneratedTrainingConfig,
): config is GeneratedTrainingConfig {
  return (
    !!config &&
    config.schema_version === 1 &&
    config.language === 'python' &&
    typeof config.source === 'string' &&
    config.source.trim().length > 0 &&
    typeof config.sha256 === 'string' &&
    /^[a-f0-9]{64}$/u.test(config.sha256)
  );
}

export function configurationConfirmed(
  config: GeneratedTrainingConfig | undefined,
  confirmedSha256: string | undefined,
) {
  return generatedConfigAvailable(config) && config.sha256 === confirmedSha256;
}

export function automaticConfigErrors(config: AutoTrainingConfig): string[] {
  const errors: string[] = [];
  if (!['pi0', 'pi05'].includes(config.model_type)) errors.push('请选择支持的模型结构。');
  if (!['full', 'lora'].includes(config.finetune_mode)) errors.push('请选择微调方式。');
  if (
    !Number.isSafeInteger(config.action_horizon) ||
    config.action_horizon < 1 ||
    config.action_horizon > 1000
  )
    errors.push('动作预测长度必须是 1–1000 的整数。');
  if (!Number.isSafeInteger(config.action_dim) || config.action_dim < 1 || config.action_dim > 4096)
    errors.push('模型动作维度必须是 1–4096 的整数。');
  return errors;
}
