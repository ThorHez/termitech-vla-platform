import type { ParameterSpec } from './types';

export interface TrainingParameterIssue {
  key: string;
  message: string;
}

/** Inherited optional settings must never turn into the old fallback value 1. */
export function trainingParameterDefaults(
  specs: Record<string, ParameterSpec>,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(specs).flatMap(([key, spec]) =>
      typeof spec.default === 'number' && Number.isFinite(spec.default)
        ? [[key, spec.default]]
        : optionalTrainingParameter(spec)
          ? []
          : [[key, 1]],
    ),
  );
}

export function inheritTrainingRecipeParameters(
  parameters: Record<string, number>,
  defaults: Record<string, number>,
  edited: ReadonlySet<string>,
): Record<string, number> {
  return {
    ...parameters,
    ...Object.fromEntries(
      Object.entries(defaults).filter(
        ([key]) => key !== 'fsdp_devices' && key !== 'num_workers' && !edited.has(key),
      ),
    ),
  };
}

export function trainingParameterGroups(specs: Record<string, ParameterSpec>) {
  const groups = [
    {
      id: 'resources',
      title: '基本训练与资源',
      keys: ['steps', 'batch_size', 'num_workers', 'fsdp_devices'],
    },
    {
      id: 'schedule',
      title: '学习率计划',
      keys: ['learning_rate', 'warmup_steps', 'decay_steps', 'decay_lr'],
    },
    { id: 'optimizer', title: '优化器与稳定性', keys: ['clip_gradient_norm', 'ema_decay'] },
    {
      id: 'checkpoints',
      title: '检查点与日志',
      keys: ['save_interval', 'keep_period', 'log_interval'],
    },
  ];
  const known = new Set(groups.flatMap((group) => group.keys));
  return groups
    .map((group) => ({
      ...group,
      entries: Object.entries(specs).filter(
        ([key]) => group.keys.includes(key) || (group.id === 'resources' && !known.has(key)),
      ),
    }))
    .filter((group) => group.entries.length || group.id === 'resources');
}

export function trainingParameterErrors(
  parameters: Record<string, number>,
  specs: Record<string, ParameterSpec>,
  gpuCount: number,
  recipeDefaults: Record<string, number> = {},
): TrainingParameterIssue[] {
  const issues: TrainingParameterIssue[] = [];
  for (const [key, spec] of Object.entries(specs)) {
    if (!Object.hasOwn(parameters, key)) {
      if (!optionalTrainingParameter(spec))
        issues.push({ key, message: `请填写${spec.label || key}。` });
      continue;
    }
    const value = parameters[key],
      min = spec.min ?? spec.minimum,
      max = spec.max ?? spec.maximum;
    if (
      !Number.isFinite(value) ||
      (spec.type === 'integer' && !Number.isInteger(value)) ||
      (min !== undefined && (spec.exclusive_min ? value <= min : value < min)) ||
      (max !== undefined && (spec.exclusive_max ? value >= max : value > max))
    ) {
      issues.push({
        key,
        message: `${spec.label || key}必须填写有效${spec.type === 'integer' ? '整数' : '数值'}${min !== undefined ? `，${spec.exclusive_min ? '大于' : '不小于'} ${min}` : ''}${max !== undefined ? `，${spec.exclusive_max ? '小于' : '不大于'} ${max}` : ''}。`,
      });
    }
  }
  // Only known explicit/inherited values participate. A blank old draft must not
  // be checked against a guessed native default from the display schema.
  const effective = { ...recipeDefaults, ...parameters };
  const finite = (key: string) => Number.isFinite(effective[key]);
  const fsdp = parameters.fsdp_devices;
  if (
    Number.isFinite(fsdp) &&
    Number.isInteger(fsdp) &&
    fsdp >= 1 &&
    Number.isInteger(gpuCount) &&
    gpuCount > 0 &&
    gpuCount % fsdp !== 0
  ) {
    issues.push({
      key: 'fsdp_devices',
      message: `FSDP 分片数必须能整除申请的 GPU 数量 ${gpuCount}；留空时跟随 GPU 数量。`,
    });
  }
  if (
    finite('warmup_steps') &&
    finite('decay_steps') &&
    effective.decay_steps <= effective.warmup_steps
  )
    issues.push({ key: 'decay_steps', message: '学习率衰减步数必须大于预热步数。' });
  if (finite('decay_lr') && finite('learning_rate') && effective.decay_lr > effective.learning_rate)
    issues.push({ key: 'decay_lr', message: '衰减结束学习率不能大于峰值学习率。' });
  return issues;
}

export function optionalTrainingParameter(spec: ParameterSpec): boolean {
  return spec.optional === true || spec.required === false;
}

/** Clearing an override restores inheritance; an incomplete required value stays invalid. */
export function editTrainingParameter(
  parameters: Record<string, number>,
  key: string,
  text: string,
  spec: ParameterSpec,
): Record<string, number> {
  const next = { ...parameters };
  if (text === '' && optionalTrainingParameter(spec)) delete next[key];
  else next[key] = text === '' ? NaN : Number(text);
  return next;
}

export function trainingParameterHint(
  key: string,
  spec: ParameterSpec,
  advanced: boolean,
  recipeDefaults: Record<string, number>,
  edited: boolean,
): string | undefined {
  const descriptions: Record<string, string> = {
    fsdp_devices: '推荐跟随申请的 GPU 数量；手动分片数须能整除申请的 GPU 数量。',
    steps: '本次训练执行的总步数。',
    batch_size: '全局批次大小，须为申请 GPU 数量的整数倍。',
    learning_rate: '学习率预热后的峰值。',
    warmup_steps: '学习率增长到峰值的预热步数。',
    decay_steps: '学习率计划的总衰减步数，必须大于预热步数。',
    decay_lr: '衰减结束时的学习率，不能大于峰值学习率。',
    clip_gradient_norm: '梯度范数裁剪阈值，必须大于 0。',
    ema_decay: '模型参数指数滑动平均的衰减系数，范围为 0 ≤ EMA < 1。',
    save_interval: '每隔多少训练步保存一次检查点。',
    keep_period: '已保存且步数能被该周期整除的检查点长期保留，另保留最新一份。',
    log_interval: '记录训练日志与指标的步数间隔。',
  };
  const source =
    advanced && Object.hasOwn(recipeDefaults, key)
      ? `配方参考值：${recipeDefaults[key]}；本次训练使用输入框中的数值。`
      : edited
        ? '当前为本次训练设置。'
        : '已填入初始值，可直接修改。';
  return [descriptions[key] || spec.hint, source].filter(Boolean).join(' ');
}
