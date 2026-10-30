import {
  decodeTrainingDraft,
  encodeTrainingDraft,
  type TrainingDraftSnapshot,
} from './trainingDraft';
import { TRAINING_TEMPLATE_PROCESSING_MARKER } from './trainingTemplate';
import { validRequestedGpuList } from './trainingGpuSelection';
import type { AutoTrainingConfig, DatasetBinding, RecipeLiteral, Run, RunRequest } from './types';

const terminalStates = new Set(['FAILED', 'CANCELLED', 'SUCCEEDED']);
const plain = (value: unknown): value is Record<string, unknown> =>
  !!value &&
  typeof value === 'object' &&
  !Array.isArray(value) &&
  [Object.prototype, null].includes(Object.getPrototypeOf(value)) &&
  Object.values(Object.getOwnPropertyDescriptors(value)).every((property) => 'value' in property);
const text = (value: unknown, max = 256): value is string =>
  typeof value === 'string' &&
  !!value.trim() &&
  value.length <= max &&
  !/[\u0000-\u001f]/.test(value);
const assetId = (value: unknown): string => {
  const id = plain(value) ? value.id : value;
  if (!text(id) || /[\\/]/.test(id) || /^[a-z]+:/i.test(id))
    throw new Error('原训练缺少有效的资产 ID，请重新选择数据集、归一化参数或初始模型。');
  return id;
};
const numericControls = (value: unknown): Record<string, number> => {
  if (!plain(value)) throw new Error('原训练参数不完整，无法载入配置。');
  const controls: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (
      /^(?:resume|overwrite|expected_plan_digest|output_dir|checkpoint_dir|attempt_id|gpu_uuids|confirmed_config_sha256|preflight_id|idempotency_key)$/.test(
        key,
      )
    )
      continue;
    if (!/^[a-z][a-z0-9_]{0,63}$/.test(key) || typeof item !== 'number' || !Number.isFinite(item))
      throw new Error('原训练包含无效参数，请手动新建训练。');
    controls[key] = item;
  }
  return controls;
};

export function retryableRun(run: Pick<Run, 'state'>): boolean {
  return terminalStates.has(run.state);
}

export function retryTrainingName(name?: string): string {
  const base =
    (name || '').replace(/(?:\s*(?:（重新训练）|\(重新训练\)))+\s*$/u, '').trim() || '训练';
  let result = '';
  for (const character of base) {
    if (result.length + character.length > 160) break;
    result += character;
  }
  return result;
}

/** Reuse settings and registered asset IDs, never launch evidence or checkpoints. */
export function retryTrainingSnapshot(run: Run): TrainingDraftSnapshot {
  if (!plain(run)) throw new Error('原训练记录无效。');
  if (!retryableRun(run)) throw new Error('只有已结束的训练可以重新配置。');
  if (![run.node_id, run.profile_id, run.template_id].every((value) => text(value)))
    throw new Error('原训练的机器、环境或训练方式记录不完整。');
  const attempt = run.attempts?.at(-1)?.code_snapshot;
  const expectation = run.code_expectation;
  const setting = (key: string): unknown => {
    for (const owner of [run, expectation, attempt])
      if (plain(owner) && Object.hasOwn(owner, key) && owner[key] !== undefined) return owner[key];
    return undefined;
  };
  const inputs: Record<string, string> = {};
  if (run.inputs !== undefined && !plain(run.inputs)) throw new Error('原训练资产记录无效。');
  for (const [role, value] of Object.entries(run.inputs || {}))
    if (/^(?:model|dataset|norm|dataset_\d+|norm_\d+)$/.test(role)) inputs[role] = assetId(value);
  if (run.template_id !== 'smoke' && !inputs.model)
    throw new Error('原训练缺少初始模型资产 ID，无法安全恢复。');
  const request: RunRequest = {
    name: retryTrainingName(run.name),
    node_id: run.node_id,
    profile_id: run.profile_id,
    template_id: run.template_id,
    inputs,
    parameters: numericControls(run.parameters),
    gpu_count: run.gpu_count ?? (run.template_id === 'smoke' ? 0 : 1),
  };
  if (
    !Number.isSafeInteger(request.gpu_count) ||
    request.gpu_count < (run.template_id === 'smoke' ? 0 : 1)
  )
    throw new Error('原训练 GPU 数量无效。');
  const requestedGpus = setting('requested_gpu_uuids');
  if (requestedGpus != null) {
    if (
      !validRequestedGpuList(requestedGpus) ||
      !requestedGpus.length ||
      requestedGpus.length !== request.gpu_count
    )
      throw new Error('原训练指定 GPU 配置无效，请手动重新选择。');
    request.requested_gpu_uuids = [...requestedGpus];
  }
  const rawBindings = setting('dataset_bindings');
  if (rawBindings != null) {
    if (!Array.isArray(rawBindings) || rawBindings.length < 1 || rawBindings.length > 64)
      throw new Error('原训练数据绑定无效。');
    request.dataset_bindings = rawBindings.map((row): DatasetBinding => {
      if (
        !plain(row) ||
        typeof row.weight !== 'number' ||
        !Number.isFinite(row.weight) ||
        row.weight <= 0
      )
        throw new Error('原训练的数据集权重无效。');
      const binding = {
        dataset_asset_id: assetId(row.dataset_asset_id),
        norm_asset_id: assetId(row.norm_asset_id),
        weight: row.weight,
      };
      if (row.processor != null && row.recipe_index !== undefined)
        throw new Error('原训练包含冲突的数据处理配置。');
      if (plain(row.processor) && text(row.processor.name) && plain(row.processor.parameters))
        return {
          ...binding,
          processor: {
            name: row.processor.name,
            parameters: row.processor.parameters as Record<string, RecipeLiteral>,
          },
        };
      if (
        Number.isSafeInteger(row.recipe_index) &&
        (row.recipe_index as number) >= 0 &&
        (row.recipe_index as number) < 64
      )
        return { ...binding, recipe_index: row.recipe_index as number };
      throw new Error('原训练的数据处理配置不完整。');
    });
  } else if (Object.keys(inputs).some((role) => /^(dataset|norm)_\d+$/.test(role))) {
    throw new Error('原训练缺少逐组数据绑定，无法确定处理器，请手动新建训练。');
  } else if (inputs.dataset && inputs.norm) {
    request.dataset_bindings = [
      { dataset_asset_id: inputs.dataset, norm_asset_id: inputs.norm, recipe_index: 0, weight: 1 },
    ];
  } else if (run.template_id !== 'smoke') {
    throw new Error('原训练缺少成对的数据集和归一化参数资产。');
  }
  if (request.dataset_bindings) request.inputs = inputs.model ? { model: inputs.model } : {};
  const rawAuto = setting('auto_config');
  const requestedEntrypoint =
    Object.hasOwn(run, 'entrypoint') && run.entrypoint !== undefined
      ? run.entrypoint
      : setting('requested_entrypoint');
  if (rawAuto != null) {
    if (
      !request.dataset_bindings?.length ||
      request.dataset_bindings.some((binding) => !('processor' in binding))
    )
      throw new Error('原自动训练缺少独立数据处理器，请手动重新配置数据处理。');
    if (text(requestedEntrypoint, 1024))
      throw new Error(
        '原训练同时使用自动配置和自定义训练入口，请在高级设置中重新确认入口后新建训练。',
      );
    if (
      !plain(rawAuto) ||
      !['pi0', 'pi05'].includes(String(rawAuto.model_type)) ||
      !['full', 'lora'].includes(String(rawAuto.finetune_mode)) ||
      !Number.isSafeInteger(rawAuto.action_horizon) ||
      (rawAuto.action_horizon as number) <= 0 ||
      !Number.isSafeInteger(rawAuto.action_dim) ||
      (rawAuto.action_dim as number) <= 0
    )
      throw new Error('原训练模型配置无效。');
    request.auto_config = {
      model_type: rawAuto.model_type,
      finetune_mode: rawAuto.finetune_mode,
      action_horizon: rawAuto.action_horizon,
      action_dim: rawAuto.action_dim,
    } as AutoTrainingConfig;
  } else {
    const definition = setting('recipe_definition');
    if (plain(definition) && text(definition.id)) {
      request.recipe_revision_id = definition.id;
      if (text(definition.base_recipe)) request.recipe = definition.base_recipe;
    }
    if (!request.recipe && text(setting('recipe'))) request.recipe = setting('recipe') as string;
    if (text(requestedEntrypoint, 1024)) request.entrypoint = requestedEntrypoint;
  }
  const stage = setting('training_stage');
  if (run.template_id === 'recap') {
    if (stage !== 'value' && stage !== 'offline_rl')
      throw new Error('原训练 ReCap 阶段记录不完整。');
    request.training_stage = stage;
    const config = setting('stage_config');
    if (config != null) request.stage_config = numericControls(config);
  }
  const snapshot: TrainingDraftSnapshot = {
    request,
    step: 0,
    advancedRecipe: run.template_id !== 'smoke' && !request.auto_config,
    editedParameters: [
      ...Object.keys(request.parameters),
      TRAINING_TEMPLATE_PROCESSING_MARKER,
      ...(request.auto_config ? ['auto_config.action_horizon'] : []),
    ],
    editedStageParameters: Object.keys(request.stage_config || {}),
    entrypointEdited: !!request.entrypoint,
  };
  // The codec bounds JSON, rejects credentials/prototype keys and makes an
  // independent deep clone of the nested per-dataset processing configuration.
  const clone = decodeTrainingDraft(encodeTrainingDraft(snapshot));
  if (!clone) throw new Error('原训练配置无法转换为新的训练草稿。');
  return clone;
}

export function sameTrainingConfiguration(
  a: TrainingDraftSnapshot,
  b: TrainingDraftSnapshot,
): boolean {
  const canonical = (value: unknown): unknown => {
    if (Array.isArray(value)) return ['array', value.map(canonical)];
    if (plain(value))
      return [
        'object',
        Object.keys(value)
          .sort()
          .filter((key) => value[key] !== undefined)
          .map((key) => [key, canonical(value[key])]),
      ];
    return [
      typeof value,
      typeof value === 'number' && !Number.isFinite(value) ? String(value) : value,
    ];
  };
  try {
    return JSON.stringify(canonical(a.request)) === JSON.stringify(canonical(b.request));
  } catch {
    return false;
  }
}
