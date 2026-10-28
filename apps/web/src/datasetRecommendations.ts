import type { DatasetProfile } from './DatasetProfileForm';
import { readDatasetProfile } from './DatasetProfileForm';
import { processorDefaults, processorParameterErrors } from './DataProcessorPicker';
import type { Asset, DataProcessorEntry, RecipeLiteral } from './types';

export interface DatasetRecommendation {
  status: 'missing_profile' | 'no_match' | 'incomplete' | 'ambiguous' | 'unique';
  candidates: { entry: DataProcessorEntry; missing: string[] }[];
  selected?: DataProcessorEntry;
  message: string;
}

export function recommendDatasetProcessor(
  asset: Asset | undefined,
  entries: DataProcessorEntry[],
  stage: 'sft' | 'value' | 'offline_rl' = 'sft',
  modelFamily?: 'standard' | 'memory' | 'hybrid',
  managedParameters: Record<string, RecipeLiteral> = {},
): DatasetRecommendation {
  const registered = readDatasetProfile(asset?.metadata?.dataset_profile);
  if (!registered)
    return {
      status: 'missing_profile',
      candidates: [],
      message: '尚未登记数据特征，可在「数据与模型」补录后自动匹配。',
    };
  const actionLabels =
    registered.field_annotations?.filter((field) => field.role === 'action') || [];
  if (actionLabels.length && actionLabels.every((field) => field.component === 'gripper'))
    return {
      status: 'incomplete',
      candidates: [],
      message: '当前仅登记了夹爪动作字段，请补充机械臂动作字段，或选择可兼容的专用处理器。',
    };
  const actions =
    registered.field_annotations?.filter(
      (field) => field.role === 'action' && field.component !== 'gripper',
    ) || [];
  if (
    actions.length &&
    (actions.some(
      (field) => !['eef', 'joint'].includes(field.representation) || field.value_mode === 'unknown',
    ) ||
      new Set(actions.map((field) => field.representation)).size !== 1 ||
      new Set(actions.map((field) => field.value_mode)).size !== 1)
  )
    return {
      status: 'incomplete',
      candidates: [],
      message:
        '原始动作字段标签尚未统一确认，请先在数据资产确认 EEF/Joint 和绝对/增量，或在本次训练中选择可兼容的专用处理器。',
    };
  const profile: DatasetProfile = actions.length
    ? {
        ...registered,
        action_representation: actions[0].representation as 'eef' | 'joint',
        action_mode: actions[0].value_mode,
      }
    : registered;
  if (profile.format !== 'lerobot')
    return { status: 'no_match', candidates: [], message: '当前独立处理器仅匹配 LeRobot 数据。' };
  const candidates: DatasetRecommendation['candidates'] = [];
  const cameraKeys = new Set(profile.cameras.map((camera) => camera.key));
  const declared = new Set([...(profile.feature_keys || []), ...cameraKeys]);
  for (const entry of entries) {
    const contract = entry.dataset_contract;
    if (
      !entry.available ||
      !entry.source ||
      !contract ||
      contract.evidence !== 'static_repack' ||
      contract.arm_mode !== profile.arm_mode ||
      contract.action_representation !== profile.action_representation ||
      contract.training_stage !== stage ||
      (modelFamily && contract.model_family && contract.model_family !== modelFamily) ||
      !contract.camera_keys.length ||
      contract.camera_keys.length !== cameraKeys.size ||
      contract.camera_keys.some((key) => !cameraKeys.has(key)) ||
      processorParameterErrors(
        processorDefaults(entry, managedParameters),
        entry,
        managedParameters,
      ).length
    )
      continue;
    if (
      contract.action_mode &&
      contract.action_mode !== 'unknown' &&
      profile.action_mode !== 'unknown' &&
      contract.action_mode !== profile.action_mode
    )
      continue;
    if (
      contract.rotation_representation &&
      contract.rotation_representation !== 'unknown' &&
      profile.rotation_representation !== 'unknown' &&
      contract.rotation_representation !== profile.rotation_representation
    )
      continue;
    const missing = contract.required_features
      .filter((key) => !declared.has(key))
      .map((key) => `数据字段 ${key}`);
    if (!modelFamily) missing.push('当前配方的模型变换类型');
    if (
      profile.action_mode !== 'unknown' &&
      (!contract.action_mode || contract.action_mode === 'unknown')
    )
      missing.push('处理器未声明动作绝对值/增量编码，需人工确认');
    if (
      profile.rotation_representation !== 'unknown' &&
      (!contract.rotation_representation || contract.rotation_representation === 'unknown')
    )
      missing.push('处理器未声明动作旋转编码，需人工确认');
    if (
      contract.action_mode &&
      contract.action_mode !== 'unknown' &&
      profile.action_mode === 'unknown'
    )
      missing.push('动作是绝对值还是增量');
    if (
      contract.rotation_representation &&
      contract.rotation_representation !== 'unknown' &&
      profile.rotation_representation === 'unknown'
    )
      missing.push('旋转编码');
    if (!contract.model_family) missing.push('模型变换兼容性');
    candidates.push({ entry, missing });
  }
  // An incomplete alternative is still an alternative. Do not silently resolve
  // ambiguity by preferring the first name, newest version or current recipe.
  if (!candidates.length)
    return {
      status: 'no_match',
      candidates,
      message: '当前目录没有声明兼容的数据处理器，可展开高级配置；旧 Agent 需更新后提供输入契约。',
    };
  if (candidates.length > 1)
    return {
      status: 'ambiguous',
      candidates,
      message: `有 ${candidates.length} 个候选处理器，请确认一次；平台不会按类名或排列顺序猜选。`,
    };
  if (candidates[0].missing.length)
    return {
      status: 'incomplete',
      candidates,
      message: '已找到候选；补全资产中的字段清单和编码后可自动应用，或展开高级配置确认。',
    };
  return {
    status: 'unique',
    candidates,
    selected: candidates[0].entry,
    message: '已根据登记数据特征匹配处理器，预检仍会核对实际数据和归一化参数。',
  };
}

export function datasetProfileFingerprint(profile: DatasetProfile | undefined) {
  return profile ? JSON.stringify(profile) : '';
}
