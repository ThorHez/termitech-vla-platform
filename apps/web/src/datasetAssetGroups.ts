import type { CatalogAssetEntry } from './modelAssetGroups';
import type { Asset } from './types';

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

/** Only group bindings whose dataset and norm both survived the current filters. */
export function groupDatasetNormAssets(entries: CatalogAssetEntry[]): AssetCatalogEntry[] {
  const norms = new Map<string, Asset>();
  for (const entry of entries)
    if (entry.kind === 'asset' && entry.asset.kind === 'NORM')
      norms.set(entry.asset.id, entry.asset);

  const groups = new Map<string, DatasetAssetGroup>();
  const assetGroups = new Map<string, DatasetAssetGroup>();
  for (const entry of entries) {
    if (entry.kind !== 'asset' || entry.asset.kind !== 'DATASET') continue;
    const dataset = entry.asset;
    const norm =
      typeof dataset.norm_asset_id === 'string' ? norms.get(dataset.norm_asset_id) : undefined;
    if (!norm || norm.node_id !== dataset.node_id) continue;
    let group = groups.get(norm.id);
    if (!group) {
      group = {
        kind: 'dataset-group',
        key: `dataset-norm:${JSON.stringify([norm.node_id, norm.id])}`,
        name: dataset.name,
        nodeId: dataset.node_id,
        datasets: [],
        norm,
        assets: [],
      };
      groups.set(norm.id, group);
      assetGroups.set(norm.id, group);
    }
    group.datasets.push(dataset);
    assetGroups.set(dataset.id, group);
  }
  for (const group of groups.values()) {
    group.name =
      group.datasets.length === 1
        ? group.datasets[0].name
        : `${group.norm.name} · ${group.datasets.length} 个数据集`;
    group.assets = [...group.datasets, group.norm];
  }

  const emitted = new Set<string>();
  const result: AssetCatalogEntry[] = [];
  for (const entry of entries) {
    const group = entry.kind === 'asset' ? assetGroups.get(entry.asset.id) : undefined;
    if (!group) {
      result.push(entry);
    } else if (!emitted.has(group.key)) {
      emitted.add(group.key);
      result.push(group);
    }
  }
  return result;
}
