import type { AssetCatalogEntry } from './datasetAssetGroups';

export const ASSET_CATALOG_PAGE_SIZE = 20;

export function paginateAssetCatalog(entries: AssetCatalogEntry[], requestedPage: number) {
  const pageSize = ASSET_CATALOG_PAGE_SIZE;
  const total = entries.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const page =
    Number.isInteger(requestedPage) && requestedPage > 0 ? Math.min(totalPages, requestedPage) : 1;
  const offset = (page - 1) * pageSize;
  const items = entries.slice(offset, offset + pageSize);
  const assetIds = [
    ...new Set(
      items.flatMap((entry) =>
        entry.kind === 'asset' ? [entry.asset.id] : entry.assets.map((asset) => asset.id),
      ),
    ),
  ];
  return {
    items,
    page,
    pageSize,
    total,
    totalPages,
    start: total ? offset + 1 : 0,
    end: offset + items.length,
    assetIds,
  };
}
