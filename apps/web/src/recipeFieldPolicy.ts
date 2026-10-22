// Keep this resource boundary aligned with vla_adapters.recipe_definitions.
const assetFields = new Set([
  'repo_id',
  'repo_ids',
  'assets',
  'assets_dir',
  'asset_id',
  'asset_ids',
  'norm_stats',
  'norm_stats_path',
  'norm_stats_dir',
  'norm_path',
  'norm_dir',
  'normalization_path',
  'normalization_dir',
  'dataset_path',
  'dataset_dir',
  'data_path',
  'data_dir',
  'rlds_data_dir',
  'filter_dict_path',
  'weight_loader',
  'weight_loaders',
  'init_weights',
  'pretrained_path',
  'pretrained_dir',
  'weights_path',
  'weight_path',
  'model_path',
  'model_dir',
  'checkpoint_path',
  'checkpoint_dir',
  'params_path',
  'params_dir',
]);

function fieldTokens(path: string): (string | number)[] | undefined {
  if (
    path.length > 512 ||
    !/^[A-Za-z][A-Za-z0-9_]*(?:(?:\.[A-Za-z][A-Za-z0-9_]*)|(?:\[(?:0|[1-9][0-9]?)\]))*$/.test(path)
  )
    return undefined;
  return (path.match(/[A-Za-z][A-Za-z0-9_]*|[0-9]+/g) || []).map((token) =>
    /^[0-9]+$/.test(token) ? Number(token) : token,
  );
}

function assetManagedTokens(tokens: (string | number)[]) {
  if (tokens.some((token) => typeof token === 'string' && assetFields.has(token))) return true;
  if ((tokens[0] === 'data' || tokens[0] === 'model') && tokens.includes('weights')) return true;
  return (
    tokens[0] === 'data' &&
    (tokens.at(-1) === 'datasets' ||
      (tokens.at(-2) === 'datasets' && typeof tokens.at(-1) === 'number'))
  );
}

export function isAssetManagedField(path: string) {
  const tokens = fieldTokens(path);
  return tokens !== undefined && assetManagedTokens(tokens);
}

function containsAssetManagedValue(
  tokens: (string | number)[],
  value: unknown,
  depth = 0,
): boolean {
  if (assetManagedTokens(tokens)) return true;
  if (depth > 4) return false;
  if (Array.isArray(value))
    return value.some((item, index) =>
      containsAssetManagedValue([...tokens, index], item, depth + 1),
    );
  if (value !== null && typeof value === 'object')
    return Object.entries(value).some(([key, item]) =>
      containsAssetManagedValue([...tokens, key], item, depth + 1),
    );
  return false;
}

export function assetManagedOverridePaths(overrides: Record<string, unknown>) {
  return Object.entries(overrides)
    .filter(([path, value]) => {
      const tokens = fieldTokens(path);
      return tokens !== undefined && containsAssetManagedValue(tokens, value);
    })
    .map(([path]) => path);
}
