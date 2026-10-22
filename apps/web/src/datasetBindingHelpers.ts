import type { DatasetBinding, RecipeDatasetSlot, RecipeDetail } from './types';

export function normalizedAssetPath(path: string) {
  return path.replace(/\/+$/u, '').replace(/\/\.\//gu, '/');
}

export function recipeDatasetSlots(detail: RecipeDetail): RecipeDatasetSlot[] {
  const fields = new Map(detail.fields.map((field) => [field.path, field]));
  const indices = [
    ...new Set(
      detail.fields.flatMap((field) => {
        const match = /^data\.datasets\[(\d+)\]/u.exec(field.path);
        return match ? [Number(match[1])] : [];
      }),
    ),
  ].sort((a, b) => a - b);
  const multi =
    indices.length > 0 || fields.get('data')?.expression.includes('MultiDataConfigFactory');
  if (multi && !indices.length) return [];
  return (multi ? indices : [0]).map((index) => {
    const prefix = multi ? `data.datasets[${index}]` : 'data';
    const factory = /([A-Za-z_][A-Za-z_0-9]*)\s*\(/u.exec(
      fields.get(prefix)?.expression || '',
    )?.[1];
    return {
      multiple: !!multi,
      recipe_index: index,
      weight: 1,
      processing: factory || '配方默认处理',
    };
  });
}

export function newDatasetBindings(slots: RecipeDatasetSlot[]): DatasetBinding[] {
  const defaults = slots.length ? slots : [{ recipe_index: 0 }];
  return defaults.map((slot) => ({
    recipe_index: slot.recipe_index,
    weight: 1,
    dataset_asset_id: '',
    norm_asset_id: '',
  }));
}

export function datasetInputLabel(role: string) {
  const match = /^(dataset|norm)_(\d+)$/u.exec(role);
  return match
    ? `${match[1] === 'dataset' ? '数据集' : '归一化参数'} ${Number(match[2]) + 1}`
    : undefined;
}
