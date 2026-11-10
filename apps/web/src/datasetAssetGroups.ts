import type { CatalogAssetEntry } from './modelAssetGroups';
import type { Asset } from './types';

/** A catalog projection only; the complete asset set remains authoritative for APIs. */
export function datasetCatalogAssets(
  assets: readonly Asset[],
  showArchived = false,
  registrantId?: string | null,
) {
  const selectionAssets = assets.filter(
    (asset) => showArchived || (asset.status || asset.state) !== 'ARCHIVED',
  );
  // Scope before grouping so a bound norm is only folded into a matching dataset.
  const scope =
    registrantId === undefined
      ? selectionAssets
      : selectionAssets.filter((asset) => (asset.registered_by_user_id ?? null) === registrantId);
  const norms = new Map(
    scope.filter((asset) => asset.kind === 'NORM').map((asset) => [asset.id, asset]),
  );
  const boundNorms = new Map<string, Asset>();
  const embeddedNormIds = new Set<string>();
  for (const dataset of scope) {
    if (dataset.kind !== 'DATASET' || !dataset.norm_asset_id) continue;
    const norm = norms.get(dataset.norm_asset_id);
    if (!norm || norm.node_id !== dataset.node_id) continue;
    boundNorms.set(dataset.id, norm);
    embeddedNormIds.add(norm.id);
  }
  return {
    assets: scope.filter((asset) => asset.kind !== 'NORM' || !embeddedNormIds.has(asset.id)),
    selectionAssets,
    boundNorms,
    embeddedNormIds,
  };
}

export interface DatasetAssetGroup {
  kind: 'dataset-group';
  key: string;
  name: string;
  nodeId: string;
  datasets: Asset[];
  norm: Asset;
  assets: Asset[];
}

export type AssetCatalogEntry = CatalogAssetEntry | DatasetAssetGroup;

/** Each filtered dataset owns a parent row, including when several share one norm. */
export function groupDatasetNormAssets(
  entries: CatalogAssetEntry[],
  boundNorms: ReadonlyMap<string, Asset>,
): AssetCatalogEntry[] {
  return entries.map((entry) => {
    if (entry.kind !== 'asset' || entry.asset.kind !== 'DATASET') return entry;
    const dataset = entry.asset;
    const norm = boundNorms.get(dataset.id);
    if (
      !norm ||
      norm.kind !== 'NORM' ||
      norm.node_id !== dataset.node_id ||
      norm.id !== dataset.norm_asset_id
    )
      return entry;
    return {
      kind: 'dataset-group',
      key: `dataset-norm:${JSON.stringify([dataset.node_id, dataset.id])}`,
      name: dataset.name,
      nodeId: dataset.node_id,
      datasets: [dataset],
      norm,
      assets: [dataset, norm],
    };
  });
}
