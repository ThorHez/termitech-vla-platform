import { ChevronDown, Database, Layers3, Server } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { Badge } from './components';
import { DatasetValidationBadge } from './DatasetValidation';
import type { DatasetAssetGroup } from './datasetAssetGroups';
import type { ModelAssetGroup } from './modelAssetGroups';
import type { Asset, Node } from './types';
import './asset-groups.css';

export function AssetGroupRows({
  group,
  node,
  selectedIds,
  eligibleIds,
  canSelect,
  onSelectionChange,
  renderAsset,
  viewDataset,
}: {
  group: ModelAssetGroup | DatasetAssetGroup;
  node?: Node;
  selectedIds: string[];
  eligibleIds: string[];
  canSelect: boolean;
  onSelectionChange: (ids: string[], checked: boolean) => void;
  renderAsset: (asset: Asset, grouped: boolean) => ReactNode;
  viewDataset?: (asset: Asset) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const contentId = useId();
  const datasetGroup = group.kind === 'dataset-group';
  const groupType = datasetGroup ? 'dataset' : 'model';
  const groupLabel = datasetGroup ? '数据集集合' : '训练模型组';
  const memberLabel = datasetGroup
    ? `${group.assets.length} 项资产`
    : `${group.assets.length} 个检查点`;
  const selectedCount = eligibleIds.filter((id) => selectedIds.includes(id)).length;
  const checked = eligibleIds.length > 0 && selectedCount === eligibleIds.length;
  const states = new Set(group.assets.map((asset) => asset.status || asset.state || 'REGISTERED'));
  const availableCount = group.assets.filter((asset) =>
    ['READY', 'AVAILABLE', 'REGISTERED'].includes(asset.status || asset.state || 'REGISTERED'),
  ).length;
  const directories = group.assets.map((asset) => asset.path.split('/').slice(0, -1));
  const common = [...directories[0]];
  for (const directory of directories.slice(1)) {
    let length = 0;
    while (length < common.length && common[length] === directory[length]) length += 1;
    common.length = length;
  }
  const directory = common.join('/');
  return (
    <>
      <tbody className={`asset-catalog-group-heading asset-${groupType}-group-heading`}>
        <tr>
          <td className="asset-selection-column">
            <input
              type="checkbox"
              aria-label={`选择同步${groupLabel} ${group.name}`}
              title="选择或取消本组当前筛选中的全部可同步资产"
              checked={checked}
              ref={(input) => {
                if (input) input.indeterminate = selectedCount > 0 && !checked;
              }}
              disabled={!canSelect || !eligibleIds.length}
              onChange={(event) => onSelectionChange(eligibleIds, event.target.checked)}
            />
          </td>
          <td>
            {datasetGroup && viewDataset ? (
              <div className="row-link asset-catalog-group-name">
                <button
                  type="button"
                  className="icon-button asset-catalog-group-disclosure"
                  aria-label={`${expanded ? '收起' : '展开'}${groupLabel} ${group.name}`}
                  aria-expanded={expanded}
                  aria-controls={contentId}
                  onClick={() => setExpanded((value) => !value)}
                >
                  <ChevronDown size={15} className={expanded ? '' : 'collapsed'} />
                </button>
                <button
                  type="button"
                  className="row-link asset-name-link asset-catalog-group-name-link"
                  aria-label={`查看数据集统计 ${group.name}`}
                  onClick={() => viewDataset(group.datasets[0])}
                >
                  <span className="row-icon blue">
                    <Database size={16} />
                  </span>
                  <span>
                    <strong>{group.name}</strong>
                    <span className="subtext">
                      {group.datasets.length} 个数据集 · 已绑定 norm_stats
                    </span>
                  </span>
                </button>
              </div>
            ) : (
              <button
                type="button"
                className="row-link asset-name-link asset-catalog-group-toggle"
                aria-label={`${expanded ? '收起' : '展开'}${groupLabel} ${group.name}`}
                aria-expanded={expanded}
                aria-controls={contentId}
                onClick={() => setExpanded((value) => !value)}
              >
                <ChevronDown size={15} className={expanded ? '' : 'collapsed'} />
                <span className={`row-icon ${datasetGroup ? 'blue' : 'teal'}`}>
                  {datasetGroup ? <Database size={16} /> : <Layers3 size={16} />}
                </span>
                <span>
                  <strong>{group.name}</strong>
                  <span className="subtext">
                    {datasetGroup
                      ? `${group.datasets.length} 个数据集 · 已绑定 norm_stats`
                      : memberLabel}
                    {!datasetGroup && group.latestStep !== undefined
                      ? ` · 最新 ${group.latestStep.toLocaleString('zh-CN')} 步`
                      : ''}
                  </span>
                </span>
              </button>
            )}
          </td>
          <td>
            <span>{datasetGroup ? '数据集 + norm_stats' : '模型集合'}</span>
            <span className="subtext">{datasetGroup ? memberLabel : '按训练任务归组'}</span>
          </td>
          <td>
            <a className="node-link" href={`#/nodes/${group.nodeId}`}>
              <Server size={13} />
              {node?.name || group.nodeId.slice(0, 10)}
            </a>
          </td>
          <td>
            {directory ? (
              <code className="path-truncate" title={directory}>
                {directory}
              </code>
            ) : (
              <span className="subtext">展开查看资产路径</span>
            )}
          </td>
          <td>
            <div className="asset-catalog-validation">
              {states.size === 1 ? (
                <Badge
                  state={[...states][0]}
                  label={datasetGroup && states.has('REGISTERED') ? '已登记' : undefined}
                />
              ) : (
                <span className="asset-catalog-group-availability">
                  可用 {availableCount} / {group.assets.length}
                </span>
              )}
              {datasetGroup && group.datasets.length === 1 && (
                <DatasetValidationBadge validation={group.datasets[0].validation} />
              )}
            </div>
          </td>
          <td>
            <button
              type="button"
              className="text-link"
              aria-label={`${expanded ? '收起' : '展开'} ${memberLabel}：${group.name}`}
              aria-expanded={expanded}
              aria-controls={contentId}
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? '收起' : '展开'}
            </button>
          </td>
        </tr>
      </tbody>
      <tbody
        className={`asset-catalog-group-items asset-${groupType}-group-items`}
        id={contentId}
        hidden={!expanded}
      >
        {group.assets.map((asset) => renderAsset(asset, true))}
      </tbody>
    </>
  );
}
