import { inferImageLayout, type ImageFeature } from './imageLayout';

export type Arm = 'left' | 'right';
export type StateKind = 'position' | 'rotation' | 'joint' | 'gripper' | 'other';
export type ActionKind = Exclude<StateKind, 'other'>;
export type VisualSlot = 'base_0_rgb' | 'left_wrist_0_rgb' | 'right_wrist_0_rgb';
export interface StateField {
  key: string;
  width: number;
  arm: Arm | 'shared';
  kind: StateKind;
  normalize: boolean;
}
export interface ActionBlock {
  arm: Arm;
  kind: ActionKind;
  width: number;
  normalize: boolean;
}
export interface ComposedCamera {
  key: string;
  slot: VisualSlot;
  kind: 'rgb' | 'depth';
  layout: '' | 'chw' | 'hwc';
  time_index: number | null;
  depth_min: number;
  depth_max: number;
}
export interface ComposedProcessingSpec {
  schema_version: 1;
  robot_model: string;
  arm_mode: 'single' | 'bimanual';
  state_representation: 'eef' | 'joint';
  state_fields: StateField[];
  state_history: number;
  action_representation: 'eef' | 'joint';
  action_key: string;
  action_layout: ActionBlock[];
  action_mode: '' | 'absolute' | 'delta';
  rotation_representation: '' | 'axis_angle' | 'quaternion' | 'rotation_6d' | 'none';
  position_unit: '' | 'm' | 'mm';
  joint_unit: '' | 'rad' | 'deg';
  coordinate_frame: string;
  gripper_range: [number, number] | null;
  cameras: ComposedCamera[];
  prompt_key: string;
  normalization: 'quantile' | 'zscore';
  model_transform: 'standard';
  aux_keys: ('value_target' | 'is_positive')[];
}
export interface ProcessingProfileSeed {
  robot_type: string;
  arm_mode: 'single' | 'bimanual';
  action_representation: 'eef' | 'joint' | 'unknown';
  action_mode: 'absolute' | 'delta' | 'unknown';
  rotation_representation: Exclude<ComposedProcessingSpec['rotation_representation'], ''> | 'unknown';
  cameras: { name: string; key: string; kind: 'rgb' | 'depth' }[];
  /** Read-only source facts for this training binding, never a saved asset composition. */
  image_features?: readonly ImageFeature[];
  feature_keys?: string[];
  action_dim?: number;
  state_dim?: number;
  field_annotations?: ProcessingFieldAnnotation[];
}

export interface ProcessingFieldAnnotation {
  key: string;
  role: 'state' | 'action';
  representation: 'eef' | 'joint' | 'gripper' | 'unknown';
  value_mode: 'absolute' | 'delta' | 'unknown';
  component: StateKind;
  arm: Arm | 'shared' | 'unknown';
  shape?: number[];
  dtype?: string;
  names?: string[];
}

const ROOT_KEYS = [
  'schema_version',
  'robot_model',
  'arm_mode',
  'state_representation',
  'state_fields',
  'state_history',
  'action_representation',
  'action_key',
  'action_layout',
  'action_mode',
  'rotation_representation',
  'position_unit',
  'joint_unit',
  'coordinate_frame',
  'gripper_range',
  'cameras',
  'prompt_key',
  'normalization',
  'model_transform',
  'aux_keys',
];
const OPTIONAL_ROOT_KEYS = [
  'position_unit', 'joint_unit', 'coordinate_frame', 'gripper_range',
  'action_mode', 'rotation_representation', 'prompt_key',
];
const STATE_KEYS = ['key', 'width', 'arm', 'kind', 'normalize'];
const ACTION_KEYS = ['arm', 'kind', 'width', 'normalize'];
const CAMERA_KEYS = ['key', 'slot', 'kind', 'layout', 'time_index', 'depth_min', 'depth_max'];
export const VISUAL_SLOTS: VisualSlot[] = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'];
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown, maximum = 256): value is string =>
  typeof value === 'string' &&
  !!value.trim() &&
  Array.from(value.trim()).length <= maximum &&
  !/[\u0000-\u001f\u007f-\u009f]/.test(value.trim());
export const compositionSchemaKey = (value: unknown): value is string =>
  text(value) &&
  !/^[\/~]/.test(value.trim()) &&
  !/[\\:()[\]{}'"`;$]/.test(value) &&
  value
    .trim()
    .split('/')
    .every((part) => part !== '' && part !== '.' && part !== '..');
const integer = (value: unknown, minimum: number, maximum: number): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= minimum && value <= maximum;
const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
const exact = (value: Record<string, unknown>, keys: string[]) =>
  Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key));
const choice = (value: unknown, options: readonly string[]) =>
  typeof value === 'string' && options.includes(value);
const blankDeclaration = (value: unknown) =>
  value == null || (typeof value === 'string' && !value.trim());

export function composedProcessingErrors(value: unknown): string[] {
  const errors: string[] = [];
  if (
    !object(value) ||
    Object.keys(value).some((key) => !ROOT_KEYS.includes(key)) ||
    ROOT_KEYS.some((key) => !OPTIONAL_ROOT_KEYS.includes(key) && !Object.hasOwn(value, key))
  )
    return ['组合处理方案字段不完整或包含不支持的字段。'];
  if (value.schema_version !== 1) errors.push('组合处理仅支持版本 1。');
  if (!text(value.robot_model, 100)) errors.push('请填写组合处理的机器人型号。');
  if (!choice(value.arm_mode, ['single', 'bimanual'])) errors.push('请选择组合处理的单臂或双臂。');
  if (!choice(value.state_representation, ['eef', 'joint'])) errors.push('请选择状态表示。');
  if (!choice(value.action_representation, ['eef', 'joint'])) errors.push('请选择动作表示。');
  if (!integer(value.state_history, 1, 64))
    errors.push('已打包状态历史长度需要是 1 到 64 的整数。');
  if (!compositionSchemaKey(value.action_key))
    errors.push('请填写真实动作字段名，不能填写文件路径或表达式。');
  if (!blankDeclaration(value.prompt_key) && !compositionSchemaKey(value.prompt_key))
    errors.push('任务文本字段需为真实字段名，不能填写文件路径或表达式；留空则自动读取。');
  if (!blankDeclaration(value.action_mode) && !choice(value.action_mode, ['absolute', 'delta']))
    errors.push('动作模式只支持绝对量或增量；可留空沿用原始数据。');
  if (!blankDeclaration(value.rotation_representation) &&
      !choice(value.rotation_representation, ['axis_angle', 'quaternion', 'rotation_6d', 'none']))
    errors.push('动作旋转编码不受支持；可留空沿用原始数据。');
  if (!blankDeclaration(value.position_unit) && !choice(value.position_unit, ['m', 'mm']))
    errors.push('位置单位只支持米或毫米；不确定时可留空。');
  if (!blankDeclaration(value.joint_unit) && !choice(value.joint_unit, ['rad', 'deg']))
    errors.push('关节单位只支持弧度或角度；不确定时可留空。');
  if (!blankDeclaration(value.coordinate_frame) && !text(value.coordinate_frame, 100))
    errors.push('动作坐标系需为不超过 100 字符的文字，不能包含控制字符。');
  if (
    value.gripper_range != null &&
    (!Array.isArray(value.gripper_range) ||
      value.gripper_range.length !== 2 ||
      !value.gripper_range.every(finite) ||
      value.gripper_range[0] >= value.gripper_range[1])
  )
    errors.push('夹爪范围需要是有限的最小值、最大值，且最小值小于最大值。');
  if (!choice(value.normalization, ['quantile', 'zscore'])) errors.push('请选择归一化方式。');
  if (value.model_transform !== 'standard')
    errors.push('组件组合目前只支持标准 Pi0/Pi05 模型变换。');
  if (
    !Array.isArray(value.aux_keys) ||
    value.aux_keys.length > 2 ||
    !value.aux_keys.every((key) => choice(key, ['value_target', 'is_positive'])) ||
    new Set(value.aux_keys).size !== value.aux_keys.length
  )
    errors.push('辅助监督字段只支持 value_target 和 is_positive，且不能重复。');

  if (
    !Array.isArray(value.state_fields) ||
    value.state_fields.length < 1 ||
    value.state_fields.length > 32
  )
    errors.push('状态拼接需要 1 到 32 个字段。');
  else {
    const keys = new Set<string>();
    let width = 0;
    value.state_fields.forEach((field, index) => {
      if (!object(field) || !exact(field, STATE_KEYS)) {
        errors.push(`状态字段 ${index + 1} 的属性不完整。`);
        return;
      }
      if (!compositionSchemaKey(field.key))
        errors.push(`请填写状态字段 ${index + 1} 的真实键名，不能填写文件路径或表达式。`);
      else {
        if (keys.has(field.key.trim())) errors.push('状态字段名不能重复。');
        keys.add(field.key.trim());
      }
      if (!integer(field.width, 1, 128))
        errors.push(`状态字段 ${index + 1} 的宽度需要是 1 到 128 的整数。`);
      else width += field.width;
      if (
        !choice(field.arm, ['left', 'right', 'shared']) ||
        (value.arm_mode === 'single' && field.arm === 'right')
      )
        errors.push(`状态字段 ${index + 1} 的机械臂归属无效。`);
      if (
        !choice(field.kind, ['position', 'rotation', 'joint', 'gripper', 'other']) ||
        typeof field.normalize !== 'boolean'
      )
        errors.push(`状态字段 ${index + 1} 的类型或归一化开关无效。`);
    });
    if (width > 512) errors.push('每个时刻拼接的状态总宽度不能超过 512。');
  }
  if (
    !Array.isArray(value.action_layout) ||
    value.action_layout.length < 1 ||
    value.action_layout.length > 16
  )
    errors.push('动作布局需要 1 到 16 个分块。');
  else {
    const arms: Arm[] = value.arm_mode === 'bimanual' ? ['left', 'right'] : ['left'];
    const blocks = new Map<string, number>();
    let width = 0;
    value.action_layout.forEach((block, index) => {
      if (!object(block) || !exact(block, ACTION_KEYS)) {
        errors.push(`动作分块 ${index + 1} 的属性不完整。`);
        return;
      }
      if (!choice(block.arm, arms)) errors.push(`动作分块 ${index + 1} 的机械臂归属无效。`);
      if (
        !choice(block.kind, ['position', 'rotation', 'joint', 'gripper']) ||
        typeof block.normalize !== 'boolean'
      )
        errors.push(`动作分块 ${index + 1} 的类型或归一化开关无效。`);
      if (!integer(block.width, 1, 128))
        errors.push(`动作分块 ${index + 1} 的宽度需要是 1 到 128 的整数。`);
      else width += block.width;
      const key = `${block.arm}:${block.kind}`;
      if (blocks.has(key)) errors.push('同一机械臂不能重复登记相同动作分块。');
      if (integer(block.width, 1, 128)) blocks.set(key, block.width);
      if (
        (value.action_representation === 'eef' && block.kind === 'joint') ||
        (value.action_representation === 'joint' &&
          (block.kind === 'position' || block.kind === 'rotation'))
      )
        errors.push('动作分块类型与所选 EEF/Joint 表示不一致。');
    });
    if (width > 512) errors.push('动作总宽度不能超过 512。');
    for (const arm of arms) {
      if (value.action_representation === 'eef') {
        const rotationWidth = { axis_angle: 3, quaternion: 4, rotation_6d: 6 }[
          String(value.rotation_representation) as 'axis_angle'
        ];
        if (blocks.get(`${arm}:position`) !== 3)
          errors.push(`${arm === 'left' ? '左' : '右'}臂 EEF 位置分块必须是 3 维。`);
        const actualRotationWidth = blocks.get(`${arm}:rotation`);
        if (!actualRotationWidth || ![3, 4, 6].includes(actualRotationWidth))
          errors.push(`${arm === 'left' ? '左' : '右'}臂 EEF 旋转分块必须是 3、4 或 6 维。`);
        if (!blankDeclaration(value.rotation_representation) &&
            (!rotationWidth || actualRotationWidth !== rotationWidth))
          errors.push(`${arm === 'left' ? '左' : '右'}臂旋转分块宽度必须与动作旋转编码一致。`);
      } else if (value.action_representation === 'joint' && !blocks.has(`${arm}:joint`))
        errors.push(`${arm === 'left' ? '左' : '右'}臂必须有 Joint 关节动作分块。`);
    }
    if (value.action_representation === 'joint' && !blankDeclaration(value.rotation_representation) &&
        value.rotation_representation !== 'none')
      errors.push('Joint 动作的旋转编码必须选择“无独立旋转”。');
  }
  if (!Array.isArray(value.cameras) || value.cameras.length < 1 || value.cameras.length > 3)
    errors.push('标准模型只支持 1 到 3 个视觉输入，请使用支持的三个视觉槽位。');
  else {
    const keys = new Set<string>(),
      slots = new Set<string>();
    value.cameras.forEach((camera, index) => {
      if (!object(camera) || !exact(camera, CAMERA_KEYS)) {
        errors.push(`组合视觉输入 ${index + 1} 的属性不完整。`);
        return;
      }
      if (!compositionSchemaKey(camera.key))
        errors.push(`请填写组合视觉输入 ${index + 1} 的真实字段名，不能填写文件路径或表达式。`);
      else {
        if (keys.has(camera.key.trim())) errors.push('组合视觉字段名不能重复。');
        keys.add(camera.key.trim());
      }
      if (!choice(camera.slot, VISUAL_SLOTS) || slots.has(String(camera.slot)))
        errors.push('模型视觉槽位必须有效且不能重复。');
      slots.add(String(camera.slot));
      if (!choice(camera.kind, ['rgb', 'depth']) || !choice(camera.layout, ['chw', 'hwc']))
        errors.push(`组合视觉输入 ${index + 1} 的图像类型或布局无效。`);
      if (camera.time_index !== null && !integer(camera.time_index, -64, 63))
        errors.push('图像时间索引需要是 -64 到 63 的整数，静态图像请留空。');
      if (
        !finite(camera.depth_min) ||
        !finite(camera.depth_max) ||
        camera.depth_min >= camera.depth_max
      )
        errors.push('深度显示范围需要有限且最小值小于最大值。');
    });
  }
  if (Array.isArray(value.state_fields) && Array.isArray(value.cameras)) {
    const keys = [...value.state_fields, ...value.cameras].flatMap((field) =>
      object(field) && typeof field.key === 'string' ? [field.key.trim()] : [],
    );
    const unique = new Set(keys);
    const promptKey = blankDeclaration(value.prompt_key) ? '' : String(value.prompt_key).trim();
    if (
      unique.size !== keys.length ||
      unique.has(String(value.action_key).trim()) ||
      unique.has(promptKey) ||
      String(value.action_key).trim() === promptKey
    )
      errors.push('状态、视觉、动作和任务字段不能互相占用。');
    if (
      Array.isArray(value.aux_keys) &&
      value.aux_keys.some(
        (key) =>
          unique.has(String(key)) ||
          key === String(value.action_key).trim() ||
          key === promptKey,
      )
    )
      errors.push('辅助字段不能与其它输入字段重名。');
  }
  return [...new Set(errors)];
}

export function normalizeComposedProcessing(spec: ComposedProcessingSpec): ComposedProcessingSpec {
  return {
    ...spec,
    robot_model: spec.robot_model.trim(),
    action_key: spec.action_key.trim(),
    action_mode: blankDeclaration(spec.action_mode) ? '' : spec.action_mode,
    rotation_representation: blankDeclaration(spec.rotation_representation) ? '' : spec.rotation_representation,
    prompt_key: spec.prompt_key == null ? '' : spec.prompt_key.trim(),
    position_unit: blankDeclaration(spec.position_unit) ? '' : spec.position_unit,
    joint_unit: blankDeclaration(spec.joint_unit) ? '' : spec.joint_unit,
    coordinate_frame: spec.coordinate_frame == null ? '' : spec.coordinate_frame.trim(),
    state_fields: spec.state_fields.map((field) => ({ ...field, key: field.key.trim() })),
    action_layout: spec.action_layout.map((block) => ({ ...block })),
    cameras: spec.cameras.map((camera) => ({ ...camera, key: camera.key.trim() })),
    gripper_range: spec.gripper_range == null ? null : [...spec.gripper_range],
    aux_keys: [...spec.aux_keys],
  };
}

export function composedProfileErrors(
  profile: ProcessingProfileSeed,
  spec: ComposedProcessingSpec,
): string[] {
  const errors: string[] = [];
  if (profile.robot_type.trim() !== spec.robot_model.trim())
    errors.push('组合方案的机器人型号与数据特征不一致。');
  if (
    profile.arm_mode !== spec.arm_mode ||
    (profile.action_representation !== 'unknown' &&
      profile.action_representation !== spec.action_representation)
  )
    errors.push('组合方案的臂数或动作表示与数据特征不一致。');
  if (spec.action_mode && profile.action_mode !== 'unknown' && profile.action_mode !== spec.action_mode)
    errors.push('组合动作模式与数据特征不一致。');
  if (
    spec.rotation_representation && profile.rotation_representation !== 'unknown' &&
    profile.rotation_representation !== spec.rotation_representation
  )
    errors.push('组合旋转编码与数据特征不一致。');
  const registered = profile.cameras.map((camera) => `${camera.key.trim()}\0${camera.kind}`).sort();
  const composed = spec.cameras.map((camera) => `${camera.key.trim()}\0${camera.kind}`).sort();
  if (JSON.stringify(registered) !== JSON.stringify(composed))
    errors.push('组合视觉输入与登记的相机字段/类型不一致。');
  if (
    profile.action_dim !== undefined &&
    profile.action_dim !== spec.action_layout.reduce((sum, block) => sum + block.width, 0)
  )
    errors.push('组合动作布局总宽度与登记动作维度不一致。');
  if (
    profile.state_dim !== undefined &&
    profile.state_dim !== spec.state_fields.reduce((sum, field) => sum + field.width, 0)
  )
    errors.push('组合状态拼接宽度与登记状态维度不一致。');
  return errors;
}

export function draftComposedProcessing(profile: ProcessingProfileSeed): ComposedProcessingSpec {
  // Unknown source facts remain incomplete; optional declarations stay blank.
  const action = profile.feature_keys?.find((key) => key === 'actions' || key === 'action') || '';
  return {
    schema_version: 1,
    robot_model: profile.robot_type,
    arm_mode: profile.arm_mode,
    state_representation: (profile.action_representation === 'unknown'
      ? ''
      : profile.action_representation) as ComposedProcessingSpec['state_representation'],
    state_fields: [{ key: '', width: 0, arm: 'shared', kind: 'other', normalize: true }],
    state_history: 1,
    action_representation: (profile.action_representation === 'unknown'
      ? ''
      : profile.action_representation) as ComposedProcessingSpec['action_representation'],
    action_key: action,
    action_layout: [
      {
        arm: 'left',
        kind: profile.action_representation === 'joint' ? 'joint' : 'position',
        width: 0,
        normalize: true,
      },
    ],
    action_mode: (profile.action_mode === 'unknown'
      ? ''
      : profile.action_mode) as ComposedProcessingSpec['action_mode'],
    rotation_representation: (profile.rotation_representation === 'unknown'
      ? ''
      : profile.rotation_representation) as ComposedProcessingSpec['rotation_representation'],
    position_unit: '',
    joint_unit: '',
    coordinate_frame: '',
    gripper_range: null,
    cameras: profile.cameras.map((camera, index) => ({
      key: camera.key,
      slot: VISUAL_SLOTS[index] || ('' as VisualSlot),
      kind: camera.kind,
      layout: inferImageLayout(profile.image_features, camera.key, camera.kind).layout || '',
      time_index: null,
      depth_min: 0,
      depth_max: 1,
    })),
    prompt_key: profile.feature_keys?.includes('task') ? 'task' : '',
    normalization: 'quantile',
    model_transform: 'standard',
    aux_keys: [],
  };
}

export function composedProcessingSummary(spec: ComposedProcessingSpec): string {
  const state = spec.state_fields.reduce((sum, field) => sum + field.width, 0),
    action = spec.action_layout.reduce((sum, block) => sum + block.width, 0);
  return `${spec.robot_model} · ${spec.arm_mode === 'single' ? '单臂' : '双臂'} · 状态 ${spec.state_representation.toUpperCase()} ${state} 维 × ${spec.state_history} · 动作 ${spec.action_representation.toUpperCase()} ${action} 维 · ${spec.cameras.length} 路视觉 · ${spec.normalization === 'quantile' ? '分位数' : 'Z-score'}归一化`;
}

export interface SourceActionLayout {
  width?: number;
  shape?: number[];
  blocks?: ActionBlock[];
  reason?: string;
}

/** Read source dimensions and only partition coordinates with complete evidence. */
export function inferSourceActionLayout(
  profile: ProcessingProfileSeed,
  actionKey: string,
): SourceActionLayout {
  const fields = (profile.field_annotations || []).filter(
    (field) => field.role === 'action' && field.key.trim() === actionKey.trim(),
  );
  if (fields.length !== 1)
    return { reason: '所选字段没有唯一的动作字段标签，无法从数据集确定宽度。' };
  const field = fields[0];
  if (!field.shape) return { reason: '尚未读取动作字段的原始形状，请先补充数据集字段信息。' };
  if (
    !Array.isArray(field.shape) ||
    !field.shape.length ||
    field.shape.length > 5 ||
    field.shape.some((size) => !Number.isSafeInteger(size) || size < 1 || size > 2 ** 31 - 1)
  )
    return { reason: '动作字段的原始形状无效，无法确定动作宽度。' };
  const shape = [...field.shape];
  if (![1, 2].includes(shape.length))
    return { shape, reason: '原始动作形状仅支持 [D] 或 [H,D]，请核对数据集字段。' };
  const width = shape.at(-1)!;
  const source = { width, shape };
  const names = field.names || [];
  const component = (name: string): ActionKind | undefined => {
    const kinds: ActionKind[] = [];
    if (/_(?:gripper|grip|finger)_/.test(name)) kinds.push('gripper');
    if (/_(?:joint\d*|joints)_/.test(name)) kinds.push('joint');
    if (/_(?:rot|rotation|quat|quaternion|rotvec|rot6d|axis_angle)_/.test(name))
      kinds.push('rotation');
    if (/_(?:pos|position|translation)_/.test(name)) kinds.push('position');
    return kinds.length === 1 ? kinds[0] : undefined;
  };
  const explicitArm = field.arm === 'left' || field.arm === 'right' ? field.arm : undefined;
  const blocks: ActionBlock[] = [];
  if (names.length) {
    if (names.length !== width || names.some((name) => typeof name !== 'string' || !name.trim()))
      return {
        ...source,
        reason: '已读到动作总宽度，但缺少完整的逐维坐标名称，无法自动确定每个分块的宽度。',
      };
    for (const name of names) {
      const token = '_' + name.toLowerCase().replace(/[^a-z0-9]+/g, '_') + '_';
      const left = /_(?:left|robot0|arm0)_/.test(token);
      const right = /_(?:right|robot1|arm1)_/.test(token);
      if (left && right)
        return { ...source, reason: '坐标名称包含冲突的左右归属，无法自动填充分块。' };
      const namedArm = left ? 'left' : right ? 'right' : undefined;
      if (namedArm && explicitArm && namedArm !== explicitArm)
        return { ...source, reason: '坐标名称与字段登记的归属冲突，请确认后再填充分块。' };
      const arm = namedArm || explicitArm || (profile.arm_mode === 'single' ? 'left' : undefined);
      const kind = component(token);
      if (!arm || !kind)
        return {
          ...source,
          reason: '坐标名称未完整说明动作类型或左右归属，无法自动确定每个分块的宽度。',
        };
      if (
        (field.component !== 'other' && field.component !== kind) ||
        (field.representation === 'joint' && (kind === 'position' || kind === 'rotation')) ||
        (field.representation === 'eef' && kind === 'joint') ||
        (field.representation === 'gripper' && kind !== 'gripper')
      )
        return { ...source, reason: '坐标名称与字段登记的表示或分量冲突，请确认后再填充分块。' };
      const previous = blocks.at(-1);
      if (previous?.arm === arm && previous.kind === kind) previous.width += 1;
      else
        blocks.push({ arm, kind, width: 1, normalize: kind !== 'rotation' && kind !== 'gripper' });
    }
  } else if (field.component !== 'other' && (explicitArm || profile.arm_mode === 'single')) {
    blocks.push({
      arm: explicitArm || 'left',
      kind: field.component,
      width,
      normalize: field.component !== 'gripper' && field.component !== 'rotation',
    });
  }
  if (!blocks.length)
    return {
      ...source,
      reason: '已读到动作总宽度，但缺少逐维坐标名称或明确的分块标签；请手动确认分块。',
    };
  if (new Set(blocks.map((block) => `${block.arm}:${block.kind}`)).size !== blocks.length)
    return {
      ...source,
      reason: '同一归属和类型的坐标分散在不同位置，当前组合不支持重排或重复分块，请手动核对。',
    };
  if (blocks.length > 16 || blocks.some((block) => block.width > 128))
    return { ...source, reason: '已识别的动作布局超出当前分块数量或单块宽度上限，请手动核对。' };
  return { ...source, blocks };
}

// Field labels describe the source. They only seed this training's editable
// draft; widths come from actual shapes, never the robot name or total width.
export function trainingCompositionDraft(profile: ProcessingProfileSeed): ComposedProcessingSpec {
  const draft = draftComposedProcessing(profile);
  const annotations = profile.field_annotations || [];
  const states = annotations.filter((field) => field.role === 'state');
  const actions = annotations.filter((field) => field.role === 'action');
  const width = (field: ProcessingFieldAnnotation) => {
    const value = field.shape?.at(-1);
    return Number.isInteger(value) && value! > 0 && value! <= 128 ? value! : 0;
  };
  const representation = (fields: ProcessingFieldAnnotation[]) => {
    const values = [
      ...new Set(
        fields
          .filter((field) => field.component !== 'gripper')
          .map((field) => field.representation)
          .filter((value): value is 'eef' | 'joint' => value === 'eef' || value === 'joint'),
      ),
    ];
    return values.length === 1 ? values[0] : undefined;
  };
  if (states.length) {
    draft.state_fields = states.map((field) => ({
      key: field.key,
      width: width(field),
      arm: field.arm === 'unknown' ? 'shared' : field.arm,
      kind: field.component,
      normalize: field.component !== 'gripper' && field.component !== 'rotation',
    }));
    const stateRepresentation = representation(states);
    if (stateRepresentation) draft.state_representation = stateRepresentation;
    const histories = states.map((field) =>
      field.shape?.length === 1 ? 1 : field.shape?.length === 2 ? field.shape[0] : undefined,
    );
    if (histories.every((value) => value !== undefined && value === histories[0]))
      draft.state_history = histories[0]!;
  }
  if (actions.length !== 1) return draft;
  const action = actions[0];
  draft.action_key = action.key;
  if (action.representation === 'eef' || action.representation === 'joint')
    draft.action_representation = action.representation;
  draft.action_mode = (
    action.value_mode === 'unknown' ? '' : action.value_mode
  ) as ComposedProcessingSpec['action_mode'];
  const inferred = inferSourceActionLayout(profile, action.key);
  if (inferred.blocks) draft.action_layout = inferred.blocks;
  if (action.representation === 'joint') draft.rotation_representation = 'none';
  return draft;
}

export function trainingCompositionProfileErrors(
  profile: ProcessingProfileSeed,
  spec: ComposedProcessingSpec,
): string[] {
  const errors: string[] = [];
  const facts = profile.field_annotations || [];
  const sourceKeys = new Set([...(profile.feature_keys || []), ...facts.map((field) => field.key)]);
  for (const field of spec.state_fields) {
    const fact = facts.find((item) => item.key === field.key);
    if (profile.field_annotations !== undefined && !fact)
      errors.push(`状态字段 ${field.key} 缺少登记标签，请回到数据资产补充。`);
    if (fact?.value_mode === 'unknown')
      errors.push(`状态字段 ${field.key} 的绝对/增量标签尚未确认，请回到数据资产补充。`);
    if (fact?.representation === 'unknown' && !['gripper', 'other'].includes(fact.component))
      errors.push(`状态字段 ${field.key} 的 EEF/Joint 标签尚未确认，请回到数据资产补充。`);
    if (sourceKeys.size && !sourceKeys.has(field.key))
      errors.push(`状态字段 ${field.key} 不在登记字段中。`);
    if (fact && fact.role !== 'state') errors.push(`字段 ${field.key} 未标记为状态字段。`);
    if (fact?.shape?.length && fact.shape.at(-1) !== field.width)
      errors.push(`状态字段 ${field.key} 宽度与原始形状不一致。`);
    if (
      fact &&
      fact.representation !== 'unknown' &&
      fact.representation !== spec.state_representation &&
      fact.component !== 'gripper'
    )
      errors.push(`状态字段 ${field.key} 的源表示与当前处理方案不一致。`);
  }
  const action = facts.find((field) => field.key === spec.action_key);
  if (profile.field_annotations !== undefined && !action)
    errors.push('所选动作字段缺少登记标签，请回到数据资产补充。');
  if (action?.representation === 'unknown')
    errors.push('动作字段的 EEF/Joint 标签尚未确认，请回到数据资产补充。');
  if (action?.representation === 'gripper' || action?.component === 'gripper')
    errors.push('所选动作字段仅包含夹爪开度，当前组合需要完整的机械臂动作字段。');
  if (sourceKeys.size && !sourceKeys.has(spec.action_key)) errors.push('动作字段不在登记字段中。');
  if (action && action.role !== 'action') errors.push('所选动作字段未标记为动作。');
  if (
    action &&
    action.representation !== 'unknown' &&
    action.representation !== spec.action_representation
  )
    errors.push('处理方案的动作表示与源字段标签不一致；当前组合不会执行 EEF/Joint 转换。');
  const mode = action?.value_mode || profile.action_mode;
  if (spec.action_mode && mode !== 'unknown' && mode !== spec.action_mode)
    errors.push('处理方案的动作绝对值/增量与源字段标签不一致。');
  if (
    action?.shape?.length &&
    action.shape.at(-1) !== spec.action_layout.reduce((sum, block) => sum + block.width, 0)
  )
    errors.push('动作分块总宽度与原始动作形状不一致。');
  for (const camera of spec.cameras) {
    if (!profile.cameras.some((source) => source.key === camera.key && source.kind === camera.kind))
      errors.push(`视觉字段 ${camera.key} 不在登记的相机字段中，或图像类型不一致。`);
  }
  return [...new Set(errors)];
}

export function updateComposedProcessingFromProfile(
  profile: ProcessingProfileSeed,
  spec: ComposedProcessingSpec,
): ComposedProcessingSpec {
  const used = new Set<VisualSlot>();
  const reserved = new Set(
    profile.cameras.flatMap((camera) => {
      const old = spec.cameras.find((item) => item.key.trim() === camera.key.trim());
      return old ? [old.slot] : [];
    }),
  );
  const cameras = profile.cameras.map((camera, index): ComposedCamera => {
    const old = spec.cameras.find((item) => item.key.trim() === camera.key.trim());
    const prior = old || spec.cameras[index];
    const slot = old
      ? old.slot
      : prior && !used.has(prior.slot) && !reserved.has(prior.slot)
        ? prior.slot
        : VISUAL_SLOTS.find((candidate) => !used.has(candidate) && !reserved.has(candidate)) ||
          ('' as VisualSlot);
    used.add(slot);
    return {
      key: camera.key,
      kind: camera.kind,
      slot,
      layout:
        inferImageLayout(profile.image_features, camera.key, camera.kind).layout || prior?.layout || '',
      time_index: prior?.time_index ?? null,
      depth_min: prior?.depth_min ?? 0,
      depth_max: prior?.depth_max ?? 1,
    };
  });
  return {
    ...spec,
    robot_model: profile.robot_type,
    arm_mode: profile.arm_mode,
    action_representation:
      profile.action_representation === 'unknown'
        ? spec.action_representation
        : profile.action_representation,
    action_mode: profile.action_mode === 'unknown' ? spec.action_mode : profile.action_mode,
    rotation_representation:
      profile.rotation_representation === 'unknown'
        ? spec.rotation_representation
        : profile.rotation_representation,
    cameras,
  };
}

export function umiV4ComposedPreset(
  profile: ProcessingProfileSeed,
  armMode: 'single' | 'bimanual',
): ComposedProcessingSpec {
  const spec = draftComposedProcessing(profile);
  const state_fields: StateField[] = [];
  const arms: Arm[] = armMode === 'single' ? ['left'] : ['left', 'right'];
  for (const [index, arm] of arms.entries()) {
    const prefix = `${armMode === 'single' ? 'observation.' : ''}robot${index}_`;
    const row = (
      suffix: string,
      width: number,
      kind: StateKind,
      normalize: boolean,
    ): StateField => ({ key: prefix + suffix, width, arm, kind, normalize });
    state_fields.push(row('eef_pos', 3, 'position', true));
    if (armMode === 'bimanual') state_fields.push(row('eef_pos_wrt_start', 3, 'position', true));
    state_fields.push(
      row('eef_rot_axis_angle', 6, 'rotation', false),
      row('eef_rot_axis_angle_wrt_start', 6, 'rotation', false),
    );
    if (armMode === 'bimanual')
      state_fields.push(
        row(`eef_pos_wrt${1 - index}`, 3, 'position', true),
        row(`eef_rot_axis_angle_wrt${1 - index}`, 6, 'rotation', false),
      );
    state_fields.push(row('gripper_width', 1, 'gripper', false));
  }
  return {
    ...spec,
    arm_mode: armMode,
    state_representation: 'eef',
    action_representation: 'eef',
    state_fields,
    state_history: 2,
    action_key: spec.action_key || 'actions',
    prompt_key: spec.prompt_key || 'task',
    rotation_representation: 'rotation_6d',
    action_layout: arms.flatMap((arm) => [
      { arm, kind: 'position' as const, width: 3, normalize: true },
      { arm, kind: 'rotation' as const, width: 6, normalize: false },
      { arm, kind: 'gripper' as const, width: 1, normalize: false },
    ]),
    cameras: spec.cameras.map((camera, index) => ({
      ...camera,
      slot: index === 0 ? 'left_wrist_0_rgb' : index === 1 ? 'right_wrist_0_rgb' : 'base_0_rgb',
    })),
  };
}
