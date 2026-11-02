import { BarChart3, LoaderCircle, RefreshCw } from 'lucide-react';
import { useState } from 'react';
import { date, useApi } from './api';
import { ErrorNotice } from './components';
import { readDatasetProfile } from './DatasetProfileForm';
import { DatasetNormalizationCharts } from './DatasetNormalizationCharts';
import type { Asset, Node } from './types';
import './dataset-statistics.css';

interface StatisticsError {
  code?: string;
  message?: string;
}

export interface DatasetStatisticsReport {
  asset_id: string;
  node_id: string;
  status: 'READY' | 'PARTIAL' | 'FAILED' | 'OFFLINE';
  checked_at?: string;
  error?: StatisticsError;
  dataset: {
    status: string;
    path?: string;
    total_frames?: number;
    total_episodes?: number;
    fps?: number;
    camera_count?: number;
    features?: { key: string; dtype: string; shape: number[] }[];
    error?: StatisticsError;
  };
  normalization: {
    status: 'READY' | 'UNBOUND' | 'FAILED' | 'UNAVAILABLE' | 'OFFLINE';
    asset_id?: string;
    path?: string;
    features?: {
      key: string;
      dimension: number;
      mean: number[];
      std: number[];
      q01?: number[];
      q99?: number[];
      min?: number[];
      max?: number[];
    }[];
    error?: StatisticsError;
  };
}

export function statisticNumber(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? Number(value.toPrecision(6)).toString()
    : '—';
}

const count = (value: unknown) =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
    ? value.toLocaleString('zh-CN')
    : '—';
const message = (error?: StatisticsError) => error?.message || error?.code;

export function DatasetFeatureComposition({
  features,
}: {
  features: NonNullable<DatasetStatisticsReport['dataset']['features']>;
}) {
  if (!features.length) return null;
  const counts = [0, 0, 0];
  for (const feature of features) {
    const dtype = feature.dtype.toLowerCase();
    const category = ['video', 'image'].includes(dtype)
      ? 0
      : /^(float|bfloat|int|uint|bool|double|half|number)/.test(dtype)
        ? 1
        : 2;
    counts[category] += 1;
  }
  const labels = ['图像字段', '数值字段', '其他字段'];
  const categories = ['images', 'numeric', 'other'];
  return (
    <div className="dataset-feature-composition">
      <div className="dataset-feature-composition-heading">
        <h4>原始字段构成</h4>
        <span>按字段数量统计 · 共 {features.length} 项</span>
      </div>
      <div
        className="dataset-feature-composition-bar"
        role="img"
        aria-label={`原始字段构成：${labels.map((label, index) => `${label} ${counts[index]} 项`).join('，')}`}
      >
        {counts.map(
          (value, index) =>
            value > 0 && (
              <span
                key={categories[index]}
                className={`dataset-feature-category ${categories[index]}`}
                style={{ width: `${(100 * value) / features.length}%` }}
                title={`${labels[index]}：${value} 项`}
              />
            ),
        )}
      </div>
      <div className="dataset-feature-composition-legend">
        {labels.map((label, index) => (
          <span key={label}>
            <i className={`dataset-feature-category ${categories[index]}`} />
            {label}
            <strong>{counts[index]}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

export function DatasetStatistics({ asset, node }: { asset: Asset; node?: Node }) {
  const offline = Boolean(node && node.status !== 'ONLINE');
  const result = useApi<DatasetStatisticsReport>(
    offline ? null : `/assets/${encodeURIComponent(asset.id)}/statistics`,
    0,
  );
  const report = result.data?.asset_id === asset.id ? result.data : undefined;
  return (
    <section className="asset-detail-section dataset-statistics" aria-label="数据集统计">
      <div className="dataset-statistics-heading">
        <h4>
          <BarChart3 size={15} />
          数据集统计
        </h4>
        <button
          type="button"
          className="text-link"
          disabled={offline || result.loading}
          onClick={result.refresh}
        >
          <RefreshCw size={13} />
          刷新统计
        </button>
      </div>
      {offline ? (
        <p className="asset-detail-muted" role="status">
          机器离线，恢复连接后可读取统计。
        </p>
      ) : result.error ? (
        <ErrorNotice error={result.error} retry={result.refresh} title="统计读取失败" />
      ) : result.loading && !report ? (
        <p className="dataset-statistics-loading" role="status">
          <LoaderCircle size={15} className="spin" />
          正在读取已有统计…
        </p>
      ) : report ? (
        <DatasetStatisticsView
          key={`${asset.id}:${asset.norm_asset_id || ''}`}
          asset={asset}
          report={report}
        />
      ) : null}
    </section>
  );
}

export function DatasetStatisticsView({
  asset,
  report,
}: {
  asset: Asset;
  report: DatasetStatisticsReport;
}) {
  const [featureKey, setFeatureKey] = useState('');
  const [page, setPage] = useState(0);
  const fields = report.normalization.features || [];
  const feature = fields.find((field) => field.key === featureKey) || fields[0];
  const dimensions =
    feature && Number.isSafeInteger(feature.dimension)
      ? Math.max(0, Math.min(feature.dimension, 4096))
      : 0;
  const pageSize = 32;
  const pageCount = Math.max(1, Math.ceil(dimensions / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  const start = currentPage * pageSize;
  const profile = readDatasetProfile(asset.metadata?.dataset_profile);
  const fpsConflict =
    profile?.fps !== undefined &&
    report.dataset.fps !== undefined &&
    profile.fps !== report.dataset.fps;
  const normMatches = report.normalization.asset_id === (asset.norm_asset_id || undefined);
  return (
    <div>
      {message(report.error) && (
        <p className="asset-detail-warning" role="status">
          {message(report.error)}
        </p>
      )}
      {message(report.dataset.error) && message(report.dataset.error) !== message(report.error) && (
        <p className="asset-detail-warning" role="status">
          {message(report.dataset.error)}
        </p>
      )}
      <dl className="dataset-statistics-summary">
        <div>
          <dt>总帧数</dt>
          <dd>{count(report.dataset.total_frames)}</dd>
        </div>
        <div>
          <dt>轨迹数</dt>
          <dd>{count(report.dataset.total_episodes)}</dd>
        </div>
        <div>
          <dt>元信息频率</dt>
          <dd>
            {report.dataset.fps === undefined ? '—' : `${statisticNumber(report.dataset.fps)} Hz`}
          </dd>
        </div>
        <div>
          <dt>图像字段数</dt>
          <dd>{count(report.dataset.camera_count)}</dd>
        </div>
      </dl>
      {fpsConflict && (
        <p className="asset-detail-warning">
          登记频率为 {profile!.fps} Hz，元信息为 {report.dataset.fps} Hz；训练沿用登记配置。
        </p>
      )}
      {!!report.dataset.features?.length && (
        <>
          <DatasetFeatureComposition features={report.dataset.features} />
          <details className="dataset-statistics-features">
            <summary>原始字段（{report.dataset.features.length} 项）</summary>
            <div className="table-scroll">
              <table aria-label="数据集原始字段">
                <thead>
                  <tr>
                    <th>字段</th>
                    <th>类型</th>
                    <th>形状</th>
                  </tr>
                </thead>
                <tbody>
                  {report.dataset.features.map((field) => (
                    <tr key={field.key}>
                      <td>
                        <code>{field.key}</code>
                      </td>
                      <td>{field.dtype}</td>
                      <td>{field.shape.join(' × ') || '标量'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
      <div className="dataset-statistics-norm-heading">
        <h4>归一化统计 · norm_stats</h4>
        {normMatches && fields.length > 0 && report.normalization.status === 'READY' && (
          <label>
            <span>统计字段</span>
            <select
              aria-label="统计字段"
              value={feature?.key || ''}
              onChange={(event) => {
                setFeatureKey(event.target.value);
                setPage(0);
              }}
            >
              {fields.map((field) => (
                <option key={field.key} value={field.key}>
                  {field.key === 'state'
                    ? '状态 · state'
                    : field.key === 'actions'
                      ? '动作 · actions'
                      : field.key}
                  （{field.dimension} 维）
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {!normMatches ? (
        <p className="asset-detail-warning" role="status">
          norm_stats 绑定已变化，请刷新统计。
        </p>
      ) : report.normalization.status === 'UNBOUND' ? (
        <p className="asset-detail-muted">未绑定 norm_stats，暂时只展示数据集元信息。</p>
      ) : report.normalization.status !== 'READY' || !feature ? (
        <p className="asset-detail-warning" role="status">
          {message(report.normalization.error) || '归一化统计暂不可读取。'}
        </p>
      ) : (
        <>
          <p className="asset-detail-muted">
            已有统计共 {dimensions} 维，按 norm_stats 中的维度顺序显示。
          </p>
          {pageCount > 1 && (
            <div className="dataset-statistics-pagination">
              <span>
                第 {start + 1}–{Math.min(dimensions, start + pageSize)} 维 / 共 {dimensions} 维
              </span>
              <button
                type="button"
                className="button outline small"
                disabled={currentPage === 0}
                onClick={() => setPage(currentPage - 1)}
              >
                上一页
              </button>
              <button
                type="button"
                className="button outline small"
                disabled={currentPage >= pageCount - 1}
                onClick={() => setPage(currentPage + 1)}
              >
                下一页
              </button>
            </div>
          )}
          <DatasetNormalizationCharts
            key={`${feature.key}:${start}`}
            feature={feature}
            start={start}
            count={Math.min(pageSize, dimensions - start)}
          />
          <details className="dataset-statistics-table">
            <summary>查看逐维明细（{dimensions} 维）</summary>
            <div className="table-scroll">
              <table className="dataset-statistics-values" aria-label="归一化逐维统计">
                <thead>
                  <tr>
                    <th>维度</th>
                    <th>均值 mean</th>
                    <th>标准差 std</th>
                    <th>1% 分位数 q01</th>
                    <th>99% 分位数 q99</th>
                    {feature.min && <th>最小值 min</th>}
                    {feature.max && <th>最大值 max</th>}
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: Math.min(pageSize, dimensions - start) }, (_, offset) => {
                    const index = start + offset;
                    return (
                      <tr key={index}>
                        <td>{index}</td>
                        <td>{statisticNumber(feature.mean[index])}</td>
                        <td>{statisticNumber(feature.std[index])}</td>
                        <td>{statisticNumber(feature.q01?.[index])}</td>
                        <td>{statisticNumber(feature.q99?.[index])}</td>
                        {feature.min && <td>{statisticNumber(feature.min[index])}</td>}
                        {feature.max && <td>{statisticNumber(feature.max[index])}</td>}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
      <details className="dataset-statistics-sources">
        <summary>统计来源{report.checked_at ? ` · ${date(report.checked_at)}` : ''}</summary>
        {report.dataset.path && (
          <p>
            <span>数据集元信息</span>
            <code>{report.dataset.path}</code>
          </p>
        )}
        {report.normalization.path && (
          <p>
            <span>归一化参数</span>
            <code>{report.normalization.path}</code>
          </p>
        )}
      </details>
    </div>
  );
}
