import type { Asset, Run } from './types';

export interface ModelAssetGroup {
  kind: 'model-group';
  key: string;
  name: string;
  nodeId: string;
  runId?: string;
  assets: Asset[];
  latest: Asset;
  latestStep?: number;
}

export type CatalogAssetEntry = { kind: 'asset'; asset: Asset } | ModelAssetGroup;

const object = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
const text = (value: unknown) =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined;

export function checkpointStep(asset: Asset): number | undefined {
  const value = object(asset.metadata)?.step;
  const step =
    typeof value === 'number'
      ? value
      : typeof value === 'string' && /^\d+$/.test(value)
        ? Number(value)
        : NaN;
  return Number.isSafeInteger(step) && step >= 0 ? step : undefined;
}

interface ModelSource {
  key: string;
  runId?: string;
  run?: Run;
  asset: Asset;
}

/** Group only checkpoints with recorded training provenance, never matching names. */
export function catalogAssetGroups(
  filteredAssets: Asset[],
  allAssets: Asset[],
  runs: Run[] = [],
): CatalogAssetEntry[] {
  const assetsById = new Map(allAssets.map((asset) => [asset.id, asset]));
  const runsById = new Map(runs.map((run) => [run.id, run]));
  const attemptRuns = new Map<string, Run>();
  for (const run of runs)
    for (const attempt of run.attempts || []) attemptRuns.set(attempt.id, run);

  const source = (asset: Asset): ModelSource | undefined => {
    let current = asset;
    const seen = new Set<string>();
    while (current.kind === 'MODEL' && !seen.has(current.id)) {
      seen.add(current.id);
      const attemptId = text(current.origin_attempt_id);
      if (attemptId) {
        const candidate = attemptRuns.get(attemptId);
        const run = candidate?.node_id === current.node_id ? candidate : undefined;
        return {
          key: JSON.stringify([
            asset.node_id,
            current.node_id,
            run ? 'run' : 'attempt',
            run?.id || attemptId,
          ]),
          runId: run?.id,
          run,
          asset: current,
        };
      }
      const metadata = object(current.metadata);
      if (metadata?.format === 'openpi-transferred') {
        const runId = text(metadata.source_run_id);
        const sourceNode = text(metadata.source_node_id);
        const run = runId ? runsById.get(runId) : undefined;
        if (runId && sourceNode && (!run || run.node_id === sourceNode))
          return {
            key: JSON.stringify([asset.node_id, sourceNode, 'run', runId]),
            runId,
            run,
            asset: current,
          };
      }
      const sync = object(metadata?.asset_sync);
      const sourceId = text(sync?.source_asset_id);
      const original = sourceId ? assetsById.get(sourceId) : undefined;
      if (
        sync?.byte_verified !== true ||
        !original ||
        original.node_id !== text(sync.source_node_id)
      )
        return undefined;
      current = original;
    }
    return undefined;
  };

  const sources = new Map<string, ModelSource>();
  const groups = new Map<string, Asset[]>();
  for (const asset of filteredAssets) {
    if (asset.kind !== 'MODEL') continue;
    const provenance = source(asset);
    if (!provenance) continue;
    sources.set(asset.id, provenance);
    const members = groups.get(provenance.key) || [];
    members.push(asset);
    groups.set(provenance.key, members);
  }

  const step = (asset: Asset) =>
    checkpointStep(asset) ?? checkpointStep(sources.get(asset.id)!.asset);
  const emitted = new Set<string>();
  const result: CatalogAssetEntry[] = [];
  for (const asset of filteredAssets) {
    const provenance = sources.get(asset.id);
    const members = provenance && groups.get(provenance.key);
    if (!provenance || !members || members.length < 2) {
      result.push({ kind: 'asset', asset });
      continue;
    }
    if (emitted.has(provenance.key)) continue;
    emitted.add(provenance.key);
    const sorted = [...members].sort((left, right) => {
      const leftStep = step(left);
      const rightStep = step(right);
      if (leftStep !== rightStep) {
        if (leftStep === undefined) return 1;
        if (rightStep === undefined) return -1;
        return rightStep - leftStep;
      }
      const time =
        (Date.parse(right.created_at || '') || 0) - (Date.parse(left.created_at || '') || 0);
      return time || members.indexOf(left) - members.indexOf(right);
    });
    const latest = sorted[0];
    const latestSource = sources.get(latest.id)!;
    const name =
      text(provenance.run?.name) ||
      text(object(latestSource.asset.metadata)?.recipe) ||
      text(latestSource.asset.name.replace(/\s*\/\s*\d+\s*$/, '')) ||
      '训练模型';
    result.push({
      kind: 'model-group',
      key: provenance.key,
      name,
      nodeId: asset.node_id,
      runId: provenance.runId,
      assets: sorted,
      latest,
      latestStep: step(latest),
    });
  }
  return result;
}
