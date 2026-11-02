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

export function copyDatasetBinding(
  first: DatasetBinding | undefined,
  defaultRecipeIndex = 0,
): DatasetBinding {
  const settings = first ? structuredClone(first) : { recipe_index: defaultRecipeIndex, weight: 1 };
  return { ...settings, dataset_asset_id: '', norm_asset_id: '' };
}

export function datasetNormalizationMode(bindings: readonly DatasetBinding[]) {
  if (bindings.length <= 1) return 'single';
  return new Set(bindings.map((binding) => binding.norm_asset_id)).size === 1 ? 'shared' : 'legacy';
}

export function sharedNormalizationAssetId(bindings: readonly DatasetBinding[]) {
  const first = bindings[0]?.norm_asset_id || '';
  return bindings.every((binding) => binding.norm_asset_id === first) ? first : '';
}

export function bindSharedNormalization(
  bindings: readonly DatasetBinding[],
  normAssetId: string,
): DatasetBinding[] {
  return bindings.map((binding) => ({ ...binding, norm_asset_id: normAssetId }));
}

export function appendDatasetBinding(
  bindings: readonly DatasetBinding[],
  defaultRecipeIndex = 0,
): DatasetBinding[] {
  const next = copyDatasetBinding(bindings[0], defaultRecipeIndex);
  // A new multi-source selection shares its first row's explicitly chosen
  // statistics. Existing distinct row selections stay intact until switched.
  if (bindings.length === 1 || datasetNormalizationMode(bindings) === 'shared')
    next.norm_asset_id = sharedNormalizationAssetId(bindings);
  return [...bindings, next];
}

export function selectDatasetBinding(
  binding: DatasetBinding,
  selection: Pick<DatasetBinding, 'dataset_asset_id' | 'norm_asset_id'>,
  defaultRecipeIndex: number,
  preserveProcessing = false,
  sharedNormAssetId?: string,
): DatasetBinding {
  const { processor, recipe_index: recipeIndex, ...rest } = binding;
  const selected = {
    ...rest,
    ...selection,
    ...(sharedNormAssetId === undefined ? {} : { norm_asset_id: sharedNormAssetId }),
  };
  // A copied row has no asset yet. Its first selection supplies the identity
  // without replacing the processing settings that were just copied.
  if (preserveProcessing || !binding.dataset_asset_id) {
    return processor
      ? { ...selected, processor }
      : { ...selected, recipe_index: recipeIndex ?? defaultRecipeIndex };
  }
  return { ...selected, recipe_index: defaultRecipeIndex };
}

export function datasetInputLabel(role: string) {
  const match = /^(dataset|norm)_(\d+)$/u.exec(role);
  return match
    ? `${match[1] === 'dataset' ? '数据集' : '归一化参数'} ${Number(match[2]) + 1}`
    : undefined;
}
