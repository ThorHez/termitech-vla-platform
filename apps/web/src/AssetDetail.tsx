import { ArrowRightLeft, Link2, Pencil, Save, Server, Trash2 } from 'lucide-react';
import { date } from './api';
import { useAuth } from './auth';
import { canArchiveAsset } from './assetArchive';
import { Badge, Modal } from './components';
import { assetKinds } from './data';
import { datasetProfileSummary, readDatasetProfile } from './DatasetProfileForm';
import { datasetFormatProfile } from './datasetFormatTemplateData';
import { boundDatasetNorm, datasetNormSelectionError } from './datasetNormAssets';
import { DatasetValidation } from './DatasetValidation';
import { DatasetStatistics } from './DatasetStatistics';
import type { Asset, Node, Profile } from './types';
import './asset-detail.css';

export function AssetDetail({
  asset,
  node,
  assets,
  close,
  editProfile,
  bindNorm,
  sync,
  remove,
  saveTemplate,
  profiles,
  validationChanged,
  datasetChoices,
  selectDataset,
}: {
  asset: Asset;
  node?: Node;
  assets: Asset[];
  close: () => void;
  editProfile: () => void;
  bindNorm: () => void;
  sync: () => void;
  remove: () => void;
  saveTemplate?: () => void;
  profiles?: Profile[];
  validationChanged?: () => void;
  datasetChoices?: Asset[];
  selectDataset?: (id: string) => void;
}) {
  const { can, user } = useAuth();
  const state = asset.status || asset.state || 'REGISTERED';
  const archived = state === 'ARCHIVED';
  const unavailable = state === 'UNAVAILABLE';
  const writable = can('assets.write') && !archived;
  const transferable = can('assets.transfer') && !archived;
  const canSaveTemplate = can('dataset-templates.write') && !archived;
  const dataset = asset.kind === 'DATASET';
  const model = asset.kind === 'MODEL';
  const profile = dataset ? readDatasetProfile(asset.metadata?.dataset_profile) : undefined;
  const templateProfile = dataset
    ? datasetFormatProfile(asset.metadata?.dataset_profile)
    : undefined;
  const norm = dataset || model ? boundDatasetNorm(asset, assets) : undefined;
  const normError =
    dataset || model
      ? datasetNormSelectionError(asset.node_id, asset.norm_asset_id, assets)
      : undefined;
  const syncReason = unavailable
    ? '源资产当前不可用，请先确认源文件。'
    : node?.status !== 'ONLINE'
      ? '源机器离线，恢复连接后可同步。'
      : undefined;
  return (
    <Modal
      wide
      title="资产详情"
      close={close}
      className="asset-detail-modal"
      footer={
        <>
          {canArchiveAsset(user, asset, can) && (
            <button type="button" className="button danger asset-detail-remove" onClick={remove}>
              <Trash2 size={15} />
              删除登记
            </button>
          )}
          <button type="button" className="button outline" onClick={close}>
            关闭
          </button>
        </>
      }
    >
      <div className="asset-detail-heading">
        <h3>{asset.name}</h3>
        <Badge state={state} label={dataset && state === 'REGISTERED' ? '已登记' : undefined} />
      </div>
      {dataset && selectDataset && (datasetChoices?.length || 0) > 1 && (
        <label className="asset-detail-dataset-picker">
          <span>使用此归一化参数的数据集</span>
          <select
            aria-label="使用此归一化参数的数据集"
            value={asset.id}
            onChange={(event) => selectDataset(event.target.value)}
          >
            {datasetChoices!.map((choice) => (
              <option key={choice.id} value={choice.id}>
                {choice.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {(writable || transferable || canSaveTemplate) && (
        <section className="asset-detail-section asset-detail-action-section">
          <h4>资产操作</h4>
          <div className="asset-detail-actions">
            {writable && dataset && !unavailable && (
              <button type="button" className="button outline" onClick={bindNorm}>
                <Link2 size={15} />
                {asset.norm_asset_id ? '修改 norm_stats' : '绑定 norm_stats'}
              </button>
            )}
            {writable && dataset && (
              <button type="button" className="button outline" onClick={editProfile}>
                <Pencil size={15} />
                {profile ? '编辑数据特征' : '补录数据特征'}
              </button>
            )}
            {canSaveTemplate && templateProfile && saveTemplate && (
              <button type="button" className="button outline" onClick={saveTemplate}>
                <Save size={15} />
                保存为格式模板
              </button>
            )}
            {transferable && ['DATASET', 'MODEL', 'NORM'].includes(asset.kind) && (
              <button
                type="button"
                className="button outline"
                onClick={sync}
                disabled={Boolean(syncReason)}
                title={syncReason || '复制到另一台机器并登记资产'}
              >
                <ArrowRightLeft size={15} />
                同步到其他机器
              </button>
            )}
          </div>
          {syncReason && <p className="asset-detail-muted">{syncReason}</p>}
        </section>
      )}
      <dl className="asset-detail-grid">
        <div>
          <dt>资产类型</dt>
          <dd>{assetKinds[asset.kind] || asset.kind}</dd>
        </div>
        <div>
          <dt>资产版本</dt>
          <dd>{asset.version}</dd>
        </div>
        <div>
          <dt>所在机器</dt>
          <dd>
            <a className="node-link" href={`#/nodes/${encodeURIComponent(asset.node_id)}`}>
              <Server size={13} />
              {node?.name || asset.node_id}
            </a>
          </dd>
        </div>
        <div>
          <dt>登记时间</dt>
          <dd>{date(asset.created_at)}</dd>
        </div>
        <div className="asset-detail-full">
          <dt>本地完整路径</dt>
          <dd>
            <code>{asset.path}</code>
          </dd>
        </div>
      </dl>
      {dataset && (
        <DatasetStatistics
          key={`${asset.id}:${asset.norm_asset_id || ''}`}
          asset={asset}
          node={node}
        />
      )}
      {dataset && (
        <DatasetValidation
          key={asset.id}
          asset={asset}
          node={node}
          profiles={profiles}
          changed={validationChanged}
        />
      )}
      {dataset && (
        <section className="asset-detail-section">
          <h4>数据集特征</h4>
          {profile ? (
            <>
              <p>{datasetProfileSummary(profile)}</p>
              <dl className="asset-detail-grid">
                <div>
                  <dt>数据格式</dt>
                  <dd>{{ lerobot: 'LeRobot', rlds: 'RLDS', other: '其他' }[profile.format]}</dd>
                </div>
                <div>
                  <dt>采样频率</dt>
                  <dd>{profile.fps === undefined ? '未填写' : `${profile.fps} Hz`}</dd>
                </div>
                <div className="asset-detail-full">
                  <dt>任务描述</dt>
                  <dd className="asset-detail-task">{profile.task_description || '未填写'}</dd>
                </div>
                <div className="asset-detail-full">
                  <dt>视觉输入</dt>
                  <dd className="asset-detail-cameras">
                    {profile.cameras.map((camera) => (
                      <div key={camera.key}>
                        <span>
                          {camera.name} · {camera.kind === 'depth' ? '深度' : 'RGB 彩色'}
                        </span>
                        <code>{camera.key}</code>
                      </div>
                    ))}
                  </dd>
                </div>
              </dl>
            </>
          ) : (
            <p className="asset-detail-muted">尚未登记数据特征</p>
          )}
        </section>
      )}
      {(dataset || (model && asset.norm_asset_id)) && (
        <section className="asset-detail-section">
          <h4>{model ? '模型训练归一化参数 · norm_stats' : '归一化参数 · norm_stats'}</h4>
          {model && <p className="asset-detail-muted">随此模型 checkpoint 保存，供推理使用。</p>}
          {asset.norm_asset_id ? (
            <>
              <p>
                {norm?.name || asset.norm_asset_id}
                {norm && ` · ${norm.version}`}
                {norm && (
                  <span aria-label="norm_stats 状态">
                    <Badge state={norm.status || norm.state || 'REGISTERED'} />
                  </span>
                )}
              </p>
              {norm && <code className="asset-detail-norm-path">{norm.path}</code>}
              {normError && (
                <p className="asset-detail-warning" role="status">
                  {normError}
                </p>
              )}
            </>
          ) : (
            <p className="asset-detail-muted">未绑定（可选）</p>
          )}
        </section>
      )}
      {archived && (
        <p className="asset-detail-muted">此登记已删除并归档，历史训练关联和本机文件保留。</p>
      )}
    </Modal>
  );
}
