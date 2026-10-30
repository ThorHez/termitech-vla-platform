import type { ParameterSpec, RecipeDetail } from './types';
import {
  isTrainingResourceField,
  setResourceRecommendationMode,
  type TrainingResourceRecommendations,
} from './trainingResourceRecommendations';

export interface TrainingParameterCompatibility {
  emaDisabled?: boolean;
  cosineDecay?: boolean;
  gradientClip?: boolean;
}

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const RECIPE_PARAMETER_PATHS: Record<string, string> = {
  num_train_steps: 'steps',
  batch_size: 'batch_size',
  'lr_schedule.peak_lr': 'learning_rate',
  'lr_schedule.warmup_steps': 'warmup_steps',
  'lr_schedule.decay_steps': 'decay_steps',
  'lr_schedule.decay_lr': 'decay_lr',
  'optimizer.clip_gradient_norm': 'clip_gradient_norm',
  ema_decay: 'ema_decay',
  fsdp_devices: 'fsdp_devices',
  num_workers: 'num_workers',
  save_interval: 'save_interval',
  keep_period: 'keep_period',
  log_interval: 'log_interval',
};

export function recipeNumericParameterDefaults(
  detail: Pick<RecipeDetail, 'fields'>,
): Record<string, number> {
  return Object.fromEntries(
    detail.fields.flatMap((field) => {
      const key = Object.hasOwn(RECIPE_PARAMETER_PATHS, field.path)
        ? RECIPE_PARAMETER_PATHS[field.path]
        : undefined;
      return key && field.kind === 'literal' && finite(field.value) ? [[key, field.value]] : [];
    }),
  );
}

/** Recognize only the optimizer and schedule contracts visible in the recipe AST. */
export function recipeParameterCompatibility(
  detail: Pick<RecipeDetail, 'fields'>,
): TrainingParameterCompatibility {
  const result: TrainingParameterCompatibility = {};
  const root = (path: string) => detail.fields.find((field) => field.path === path);
  const schedule = root('lr_schedule');
  if (schedule?.kind === 'expression') {
    if (/\bCosineDecaySchedule\s*\(/.test(schedule.expression)) result.cosineDecay = true;
    else if (/\bRsqrtDecaySchedule\s*\(/.test(schedule.expression)) result.cosineDecay = false;
  }
  if (
    result.cosineDecay === undefined &&
    detail.fields.some((field) =>
      ['lr_schedule.decay_steps', 'lr_schedule.decay_lr'].includes(field.path),
    )
  )
    result.cosineDecay = true;
  const optimizer = root('optimizer');
  if (optimizer?.kind === 'expression') {
    if (/\bAdamW\s*\(/.test(optimizer.expression)) result.gradientClip = true;
    else if (/\bSGD\s*\(/.test(optimizer.expression)) result.gradientClip = false;
  }
  if (
    result.gradientClip === undefined &&
    detail.fields.some((field) => field.path === 'optimizer.clip_gradient_norm')
  )
    result.gradientClip = true;
  const ema = root('ema_decay');
  if (ema?.kind === 'literal' && ema.value === null) result.emaDisabled = true;
  else if (ema?.kind === 'literal' && finite(ema.value)) result.emaDisabled = false;
  else if (ema?.kind === 'expression' && ema.expression.trim() === 'None')
    result.emaDisabled = true;
  return result;
}

export function unsupportedTrainingParameters(
  compatibility: TrainingParameterCompatibility,
): string[] {
  return [
    ...(compatibility.cosineDecay === false ? ['decay_steps', 'decay_lr'] : []),
    ...(compatibility.gradientClip === false ? ['clip_gradient_norm'] : []),
  ];
}

export function initializeVisibleTrainingParameters(
  parameters: Record<string, number>,
  specs: Record<string, ParameterSpec>,
  options: {
    recipeDefaults?: Record<string, number>;
    resources: TrainingResourceRecommendations;
    compatibility?: TrainingParameterCompatibility;
    lora?: boolean;
    defaultedKeys: ReadonlySet<string>;
    edited: ReadonlySet<string>;
    phase: 'initial' | 'resolved';
  },
): {
  parameters: Record<string, number>;
  defaultedKeys: Set<string>;
  edited: Set<string>;
} {
  let next = parameters;
  const defaultedKeys = new Set(options.defaultedKeys);
  const edited = new Set(options.edited);
  const unsupported = new Set(unsupportedTrainingParameters(options.compatibility ?? {}));
  if (options.lora || options.compatibility?.emaDisabled) unsupported.add('ema_decay');
  const write = (key: string, value: number) => {
    if (Object.is(next[key], value) && Object.hasOwn(next, key)) return;
    if (next === parameters) next = { ...parameters };
    next[key] = value;
  };
  for (const key of unsupported) {
    if (
      Object.hasOwn(next, key) &&
      (defaultedKeys.has(key) || (options.phase === 'initial' && !finite(next[key])))
    ) {
      if (next === parameters) next = { ...parameters };
      delete next[key];
    }
  }
  // Fill these first so schema-only schedule defaults respect the visible values.
  const keys = [...new Set(['learning_rate', 'warmup_steps', ...Object.keys(specs)])];
  for (const key of keys) {
    const spec = specs[key];
    if (!spec || unsupported.has(key)) continue;
    const filled = finite(next[key]);
    if (isTrainingResourceField(key)) {
      if (filled) continue;
      if (options.phase === 'resolved' && edited.has(key) && !defaultedKeys.has(key)) continue;
      const recommendation = options.resources[key];
      if (!finite(recommendation)) continue;
      write(key, recommendation);
      defaultedKeys.add(key);
      setResourceRecommendationMode(edited, key, true);
      continue;
    }
    if (filled && !defaultedKeys.has(key)) continue;
    if (options.phase === 'initial' && filled) continue;
    if (options.phase === 'resolved' && !defaultedKeys.has(key) && edited.has(key)) continue;
    const recipeDefault = options.recipeDefaults?.[key];
    const fromRecipe = finite(recipeDefault);
    let value = fromRecipe
      ? recipeDefault
      : finite(spec.default)
        ? spec.default
        : key === 'learning_rate'
          ? 5e-5
          : key === 'ema_decay'
            ? 0.99
            : undefined;
    if (!finite(value)) continue;
    if (!fromRecipe && key === 'decay_lr' && finite(next.learning_rate) && next.learning_rate > 0)
      value = Math.min(value, next.learning_rate);
    if (!fromRecipe && key === 'decay_steps' && finite(next.warmup_steps) && next.warmup_steps >= 0)
      value = Math.max(value, next.warmup_steps + 1);
    write(key, value);
    // Keep provenance through replacement/removal: React can invoke an updater twice.
    defaultedKeys.add(key);
  }
  return { parameters: next, defaultedKeys, edited };
}
