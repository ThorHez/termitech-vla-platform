import type { DatasetProfile } from './DatasetProfileForm';
import { datasetProfileErrors } from './DatasetProfileForm';
import { datasetFormatProfile } from './datasetFormatTemplateData';
import { datasetNormBindingPayload, datasetNormInputError } from './datasetNormAssets';
import type { Asset } from './types';

export interface DatasetScanEntry {
  path: string;
  relative_path: string;
  name: string;
  info_sha256?: string;
  feature_signature?: string;
  summary: {
    fps?: number;
    robot_type?: string;
    total_episodes?: number;
    total_frames?: number;
    task_description?: string;
  };
  error: { code: string; message: string } | null;
  existing_asset_id?: string;
  existing_state?: string;
}

export interface DatasetScan {
  id: string;
  node_id: string;
  path: string;
  recursive: boolean;
  max_depth: number;
  state: 'QUEUED' | 'SCANNING' | 'READY' | 'FAILED';
  entries: DatasetScanEntry[];
  warnings: string[];
  truncated: boolean;
  scanned_directories: number;
  skipped_directories: number;
  error: { code: string; message: string } | null;
  expires_at: string;
}

export interface DatasetScanRow {
  entry: DatasetScanEntry;
  selected: boolean;
  name: string;
  normId: string;
  normPath: string;
  existingAssetId?: string;
}

export interface BatchDatasetResult {
  scan_id: string;
  idempotency_key: string;
  created: Asset[];
  skipped: { path: string; asset_id: string; reason: string; state?: string }[];
  created_count: number;
  skipped_count: number;
}

export function canonicalDatasetPath(path: string): string | undefined {
  if (!path.startsWith('/') || /[\\\u0000-\u001f\u007f-\u009f]/.test(path)) return undefined;
  const parts = path.split('/');
  if (parts.includes('..')) return undefined;
  return '/' + parts.filter((part) => part && part !== '.').join('/');
}

export function datasetScanScope(
  nodeId: string,
  path: string,
  recursive: boolean,
  maxDepth: number,
) {
  return JSON.stringify([
    nodeId,
    canonicalDatasetPath(path.trim()) || path.trim(),
    recursive,
    maxDepth,
  ]);
}

export function batchScanPathError(path: string): string | undefined {
  const value = path.trim();
  if (!value) return '请选择要扫描的父目录。';
  if (canonicalDatasetPath(value) === '/')
    return '请选择数据集所在目录，不能扫描文件系统根目录。';
  if (value.length > 4096 || !canonicalDatasetPath(value))
    return '父目录需要有效的本机绝对路径，不能包含 .. 或控制字符。';
}

export function scanRows(scan: DatasetScan, assets: Asset[]): DatasetScanRow[] {
  if (!Array.isArray(scan.entries) || scan.entries.length > 200)
    throw new Error('扫描结果超过本次批量登记上限，请缩小目录范围。');
  const seen = new Set<string>();
  const names = new Map<string, number>();
  for (const entry of scan.entries) names.set(entry.name, (names.get(entry.name) || 0) + 1);
  return scan.entries.map((entry) => {
    const path = canonicalDatasetPath(entry.path);
    if (!path || seen.has(path)) throw new Error('扫描返回了无效或重复的数据集路径，请重新扫描。');
    seen.add(path);
    const existing = assets.find(
      (asset) =>
        asset.node_id === scan.node_id &&
        asset.kind === 'DATASET' &&
        canonicalDatasetPath(asset.path) === path,
    );
    const name =
      names.get(entry.name)! > 1
        ? entry.relative_path.replace(/\//g, ' · ') || entry.name
        : entry.name;
    return {
      entry,
      name: Array.from(name.trim()).slice(0, 200).join(''),
      normId: '',
      normPath: '',
      existingAssetId: entry.existing_asset_id || existing?.id,
      selected: !entry.error && !entry.existing_asset_id && !existing,
    };
  });
}

export function applyBatchNorm(
  rows: DatasetScanRow[],
  normId: string,
  normPath: string,
): DatasetScanRow[] {
  return rows.map((row) =>
    row.selected && !row.entry.error && !row.existingAssetId
      ? { ...row, normId, normPath }
      : row,
  );
}

export function batchRegistrationError({
  nodeId,
  rows,
  version,
  metadataOnly,
  profile,
  assets,
}: {
  nodeId: string;
  rows: DatasetScanRow[];
  version: string;
  metadataOnly: boolean;
  profile: DatasetProfile;
  assets: Asset[];
}): string | undefined {
  const chosen = rows.filter((row) => row.selected);
  if (!chosen.length) return '请勾选至少一个未登记的数据集。';
  if (chosen.length > 200) return '一次最多登记 200 个数据集，请缩小选择范围。';
  if (
    !version.trim() ||
    Array.from(version.trim()).length > 200 ||
    /[\u0000-\u001f\u007f-\u009f]/.test(version)
  )
    return '资产版本需要 1 至 200 个字符，不能包含控制字符。';
  for (const row of chosen) {
    if (row.entry.error || row.existingAssetId)
      return '选择中包含读取失败或已登记的数据集，请重新检查。';
    if (
      !row.name.trim() ||
      Array.from(row.name.trim()).length > 200 ||
      /[\u0000-\u001f\u007f-\u009f]/.test(row.name)
    )
      return '请为每个选中的数据集填写有效名称，最多 200 个字符。';
    const normError = datasetNormInputError(nodeId, row.normId, row.normPath, assets);
    if (normError) return `数据集「${row.name.trim()}」：${normError}`;
  }
  if (!metadataOnly) {
    const error = datasetProfileErrors(profile)[0];
    if (error) return '公共数据集格式尚未完整：' + error;
    if (!datasetFormatProfile(profile)) return '公共数据集格式无效，请重新套用格式模板。';
  }
  const entries = batchRegistrationEntries(rows, version, metadataOnly, profile);
  if (
    new TextEncoder().encode(JSON.stringify({ idempotency_key: '0'.repeat(36), entries }))
      .byteLength >
    1024 * 1024
  )
    return '本批登记配置超过 1 MiB，请减少勾选的数据集数量后分批登记。';
}

export function batchRegistrationEntries(
  rows: DatasetScanRow[],
  version: string,
  metadataOnly: boolean,
  profile: DatasetProfile,
) {
  const raw = metadataOnly ? undefined : datasetFormatProfile(profile);
  if (!metadataOnly && !raw) throw new Error('数据集格式无效，请重新选择模板。');
  return rows
    .filter((row) => row.selected)
    .map((row) => ({
      path: row.entry.path,
      name: row.name.trim(),
      version: version.trim(),
      ...(raw ? { profile: structuredClone(raw) } : {}),
      ...datasetNormBindingPayload(row.normId, row.normPath),
    }));
}

export function batchIdempotency(
  previous: { fingerprint: string; key: string } | undefined,
  entries: unknown,
  makeKey: () => string,
) {
  const fingerprint = JSON.stringify(entries);
  return previous?.fingerprint === fingerprint ? previous : { fingerprint, key: makeKey() };
}
