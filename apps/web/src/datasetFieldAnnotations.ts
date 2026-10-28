export interface DatasetFieldAnnotation {
  key: string;
  role: 'state' | 'action';
  representation: 'eef' | 'joint' | 'gripper' | 'unknown';
  value_mode: 'absolute' | 'delta' | 'unknown';
  component: 'position' | 'rotation' | 'joint' | 'gripper' | 'other';
  arm: 'left' | 'right' | 'shared' | 'unknown';
  shape?: number[];
  dtype?: string;
  names?: string[];
  horizon?: number;
}

export function sourceActionHorizon(shape: unknown): number | undefined {
  if (
    !Array.isArray(shape) ||
    ![1, 2].includes(shape.length) ||
    shape.some((size) => !Number.isSafeInteger(size) || size < 1 || size > 2 ** 31 - 1)
  )
    return undefined;
  return shape.length === 1 ? 1 : shape[0];
}

const text = (value: unknown, maximum: number) =>
  typeof value === 'string' &&
  !!value.trim() &&
  Array.from(value.trim()).length <= maximum &&
  !/[\u0000-\u001f\u007f-\u009f]/.test(value);

export function fieldAnnotationErrors(value: unknown): string[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 256) return ['字段标签最多 256 项。'];
  const keys = new Set<string>();
  const allowed = new Set([
    'key',
    'role',
    'representation',
    'value_mode',
    'component',
    'arm',
    'shape',
    'dtype',
    'names',
    'horizon',
  ]);
  for (const [index, row] of value.entries()) {
    if (
      !row ||
      typeof row !== 'object' ||
      Array.isArray(row) ||
      Object.keys(row).some((key) => !allowed.has(key)) ||
      !text(row.key, 256) ||
      !['state', 'action'].includes(row.role) ||
      !['eef', 'joint', 'gripper', 'unknown'].includes(row.representation) ||
      !['absolute', 'delta', 'unknown'].includes(row.value_mode) ||
      !['position', 'rotation', 'joint', 'gripper', 'other'].includes(row.component) ||
      !['left', 'right', 'shared', 'unknown'].includes(row.arm)
    )
      return [`请完整填写字段标签 ${index + 1}。`];
    if (row.representation === 'gripper' && row.component !== 'gripper')
      return [`字段标签 ${index + 1} 的夹爪开度表示必须使用夹爪分量。`];
    if (keys.has(row.key.trim())) return ['同一个原始字段只能登记一组标签。'];
    keys.add(row.key.trim());
    if (
      row.horizon !== undefined &&
      (row.role !== 'action' ||
        !Number.isSafeInteger(row.horizon) ||
        row.horizon < 1 ||
        row.horizon > 2 ** 31 - 1)
    )
      return [`字段标签 ${index + 1} 的 horizon 仅用于动作字段，且必须为正整数。`];
    if (
      row.horizon !== undefined &&
      row.shape !== undefined &&
      row.horizon !== sourceActionHorizon(row.shape)
    )
      return [`字段标签 ${index + 1} 的 horizon 与数据原始形状不一致。`];
    if (
      row.shape !== undefined &&
      (!Array.isArray(row.shape) ||
        !row.shape.length ||
        row.shape.length > 5 ||
        row.shape.some(
          (n: unknown) =>
            !Number.isSafeInteger(n) || (n as number) <= 0 || (n as number) > 2 ** 31 - 1,
        ))
    )
      return [`字段标签 ${index + 1} 的原始形状无效。`];
    if (row.dtype !== undefined && !text(row.dtype, 64))
      return [`字段标签 ${index + 1} 的原始类型无效。`];
    if (
      row.names !== undefined &&
      (!Array.isArray(row.names) ||
        row.names.length > 512 ||
        row.names.some((name: unknown) => !text(name, 256)))
    )
      return [`字段标签 ${index + 1} 的坐标名称无效。`];
  }
  return [];
}

export function normalizedFieldAnnotations(value: DatasetFieldAnnotation[]) {
  return value.map((row) => ({
    ...row,
    key: row.key.trim(),
    ...(row.shape ? { shape: [...row.shape] } : {}),
    ...(row.names ? { names: row.names.map((name) => name.trim()) } : {}),
    ...(row.dtype ? { dtype: row.dtype.trim() } : {}),
    ...(row.role === 'action' && (row.horizon ?? sourceActionHorizon(row.shape)) !== undefined
      ? { horizon: row.horizon ?? sourceActionHorizon(row.shape) }
      : {}),
  }));
}
