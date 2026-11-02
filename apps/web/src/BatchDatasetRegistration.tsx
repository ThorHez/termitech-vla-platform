import { Fragment, useEffect, useRef, useState } from 'react';
import { Check, CheckCircle2, Database, FolderSearch, LoaderCircle } from 'lucide-react';
import { api } from './api';
import { useAuth } from './auth';
import { ErrorNotice, Field, Modal } from './components';
import { useData } from './data';
import { DatasetFormatTemplates } from './DatasetFormatTemplates';
import { DatasetProfileForm, emptyDatasetProfile } from './DatasetProfileForm';
import { DatasetNormField } from './DatasetNormBinding';
import { datasetNormInputError } from './datasetNormAssets';
import { PathAutocomplete } from './PathAutocomplete';
import { DatasetValidationBadge } from './DatasetValidation';
import {
  applyBatchNorm,
  batchIdempotency,
  batchRegistrationEntries,
  batchRegistrationError,
  batchScanPathError,
  datasetScanScope,
  scanRows,
  type BatchDatasetResult,
  type DatasetScan,
  type DatasetScanRow,
} from './batchDatasetRegistrationData';
import './batch-dataset-registration.css';

export function BatchDatasetRegistration({
  initialNode = '',
  close,
}: {
  initialNode?: string;
  close: () => void;
}) {
  const { user, can } = useAuth();
  if (!can('assets.write') || !user?.id)
    return (
      <Modal title="批量登记数据集" close={close}>
        <p>请使用具有资产编辑权限的个人账户登录后，批量登记数据集。</p>
      </Modal>
    );
  return (
    <BatchDatasetRegistrationContent
      key={user.id}
      owner={user.id}
      initialNode={initialNode}
      close={close}
    />
  );
}

function BatchDatasetRegistrationContent({
  owner,
  initialNode,
  close,
}: {
  owner: string;
  initialNode: string;
  close: () => void;
}) {
  const { user, can } = useAuth();
  const { nodes, assets, refresh } = useData();
  const [nodeId, setNodeId] = useState(initialNode);
  const [path, setPath] = useState('');
  const [recursive, setRecursive] = useState(true);
  const [maxDepth, setMaxDepth] = useState(1);
  const [scan, setScan] = useState<DatasetScan>();
  const [rows, setRows] = useState<DatasetScanRow[]>([]);
  const [version, setVersion] = useState('v1');
  const [metadataOnly, setMetadataOnly] = useState(true);
  const [profile, setProfile] = useState(emptyDatasetProfile);
  const [normId, setNormId] = useState('');
  const [normPath, setNormPath] = useState('');
  const [busy, setBusy] = useState<'scan' | 'register'>();
  const [error, setError] = useState<Error>();
  const [result, setResult] = useState<BatchDatasetResult>();
  const operation = useRef<AbortController | undefined>(undefined);
  const mounted = useRef(true);
  const attempt = useRef<{ fingerprint: string; key: string } | undefined>(undefined);
  const pending = useRef(false);
  const scope = datasetScanScope(nodeId, path, recursive, maxDepth);
  const current = useRef({ scope, owner, allowed: true });
  current.current = { scope, owner: user?.id || '', allowed: can('assets.write') };
  const node = nodes.data?.find((item) => item.id === nodeId);
  const selected = rows.filter((row) => row.selected);
  const eligible = rows.filter((row) => !row.entry.error && !row.existingAssetId);
  const selectedSignatures = new Set(
    selected.map((row) => row.entry.feature_signature).filter(Boolean),
  );
  const registrationError = batchRegistrationError({
    nodeId,
    rows,
    version,
    metadataOnly,
    profile,
    assets: assets.data || [],
  });
  const sharedNormError = datasetNormInputError(nodeId, normId, normPath, assets.data || []);
  const pathError = batchScanPathError(path);
  const headers = { 'X-Dataset-Batch-User': owner };

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      operation.current?.abort();
    };
  }, []);

  const active = (abort: AbortController, expectedScope: string) =>
    mounted.current &&
    !abort.signal.aborted &&
    operation.current === abort &&
    current.current.scope === expectedScope &&
    current.current.owner === owner &&
    current.current.allowed;
  const invalidate = () => {
    operation.current?.abort();
    operation.current = undefined;
    setBusy(undefined);
    setScan(undefined);
    setRows([]);
    setResult(undefined);
    setError(undefined);
    attempt.current = undefined;
  };
  const safeClose = () => {
    if (pending.current) return;
    operation.current?.abort();
    close();
  };
  const pause = (abort: AbortController) =>
    new Promise<void>((resolve, reject) => {
      const stopped = () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      };
      const timer = setTimeout(() => {
        abort.signal.removeEventListener('abort', stopped);
        resolve();
      }, 1500);
      abort.signal.addEventListener('abort', stopped, { once: true });
      if (abort.signal.aborted) stopped();
    });
  const scanDirectory = async () => {
    if (busy || !nodeId || pathError || !current.current.allowed || current.current.owner !== owner)
      return;
    invalidate();
    const expectedScope = scope;
    const abort = new AbortController();
    operation.current = abort;
    setBusy('scan');
    try {
      let next = await api<DatasetScan>(`/nodes/${encodeURIComponent(nodeId)}/dataset-scans`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ path: path.trim(), recursive, max_depth: maxDepth }),
        signal: AbortSignal.any([abort.signal, AbortSignal.timeout(20000)]),
      });
      while (active(abort, expectedScope)) {
        if (
          next.node_id !== nodeId ||
          datasetScanScope(next.node_id, next.path, next.recursive, next.max_depth) !==
            expectedScope
        )
          throw new Error('扫描返回的目录范围不一致，请重新扫描。');
        setScan(next);
        if (next.state === 'READY') {
          setRows(scanRows(next, assets.data || []));
          return;
        }
        if (next.state === 'FAILED')
          throw new Error(
            next.error
              ? `${next.error.code}: ${next.error.message}`
              : '扫描失败，请检查目录访问权限后重试。',
          );
        if (!['QUEUED', 'SCANNING'].includes(next.state))
          throw new Error('扫描状态无效，请重新扫描。');
        await pause(abort);
        next = await api<DatasetScan>(
          `/nodes/${encodeURIComponent(nodeId)}/dataset-scans/${encodeURIComponent(next.id)}`,
          {
            headers,
            signal: AbortSignal.any([abort.signal, AbortSignal.timeout(20000)]),
          },
        );
      }
    } catch (cause) {
      if (active(abort, expectedScope))
        setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      if (active(abort, expectedScope)) setBusy(undefined);
    }
  };

  const register = async () => {
    if (
      pending.current ||
      busy ||
      registrationError ||
      !scan ||
      scan.state !== 'READY' ||
      !current.current.allowed ||
      current.current.owner !== owner
    )
      return;
    const entries = batchRegistrationEntries(rows, version, metadataOnly, profile);
    const payload = { scan_id: scan.id, entries };
    const identity = batchIdempotency(attempt.current, payload, () => crypto.randomUUID());
    attempt.current = identity;
    const abort = new AbortController();
    operation.current?.abort();
    operation.current = abort;
    const expectedScope = scope;
    pending.current = true;
    setBusy('register');
    setError(undefined);
    try {
      const saved = await api<BatchDatasetResult>(
        `/nodes/${encodeURIComponent(nodeId)}/dataset-scans/${encodeURIComponent(scan.id)}/register`,
        {
          method: 'POST',
          headers,
          body: JSON.stringify({ idempotency_key: identity.key, entries }),
          signal: AbortSignal.any([abort.signal, AbortSignal.timeout(60000)]),
        },
      );
      if (!active(abort, expectedScope)) return;
      if (
        saved.scan_id !== scan.id ||
        saved.idempotency_key !== identity.key ||
        !Array.isArray(saved.created) ||
        !Array.isArray(saved.skipped)
      )
        throw new Error('登记返回结果不一致，请保留当前选择并重试。');
      setResult(saved);
      refresh();
    } catch (cause) {
      if (active(abort, expectedScope))
        setError(cause instanceof Error ? cause : new Error(String(cause)));
    } finally {
      pending.current = false;
      if (active(abort, expectedScope)) setBusy(undefined);
    }
  };

  return (
    <Modal
      wide
      title="批量登记数据集"
      subtitle="扫描父目录中的 LeRobot 数据集，预览后统一登记。"
      close={safeClose}
      className="batch-dataset-modal"
      footer={
        <>
          <button
            type="button"
            className="button outline"
            disabled={busy === 'register'}
            onClick={safeClose}
          >
            {result ? '完成' : '关闭'}
          </button>
          {!result && scan?.state === 'READY' && (
            <button
              type="button"
              className="button primary"
              disabled={Boolean(busy) || Boolean(registrationError)}
              onClick={() => void register()}
            >
              {busy === 'register' ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <Check size={16} />
              )}
              登记 {selected.length} 个数据集
            </button>
          )}
        </>
      }
    >
      <ErrorNotice error={error} title="批量登记未完成" />
      {result ? (
        <div className="batch-dataset-result" role="status">
          <CheckCircle2 size={30} />
          <h3>
            已登记 {result.created.length} 个数据集，跳过 {result.skipped.length} 个已有登记
          </h3>
          <p>资产已登记，基础校验在后台执行，可在资产列表与详情查看结果。训练组合配置仍需在训练向导中校验。</p>
          <ul>
            {result.created.map((asset) => (
              <li key={asset.id}>
                <Database size={14} />
                <span>{asset.name}</span>
                <code>{asset.path}</code>
                <DatasetValidationBadge validation={(assets.data?.find((current) => current.id === asset.id) || asset).validation} />
              </li>
            ))}
          </ul>
          {!!result.skipped.length && (
            <details>
              <summary>查看跳过的项目</summary>
              <ul>
                {result.skipped.map((item) => (
                  <li key={item.path}>
                    <code>{item.path}</code>
                    <span>{item.reason}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ) : (
        <>
          <section className="batch-dataset-source" aria-label="扫描目录">
            <Field label="所在机器" required>
              <select
                aria-label="批量登记所在机器"
                disabled={Boolean(busy)}
                value={nodeId}
                onChange={(event) => {
                  invalidate();
                  setNodeId(event.target.value);
                  setNormId('');
                  setNormPath('');
                }}
              >
                <option value="">选择机器</option>
                {nodes.data?.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                    {item.status === 'ONLINE' ? '' : ' · Agent 离线'}
                  </option>
                ))}
              </select>
            </Field>
            <PathAutocomplete
              node={node}
              value={path}
              label="数据集父目录"
              placeholder="例如 /data/datasets"
              disabled={Boolean(busy) || !nodeId}
              onChange={(value) => {
                invalidate();
                setPath(value);
              }}
              hint="通过已保存的 SSH 连接读取 meta/info.json；不扫描样本或视频内容。"
            />
            <div className="batch-dataset-scan-options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={recursive}
                  disabled={Boolean(busy)}
                  onChange={(event) => {
                    invalidate();
                    setRecursive(event.target.checked);
                  }}
                />
                包含子目录
              </label>
              <Field label="扫描深度">
                <input
                  type="number"
                  aria-label="扫描深度"
                  min={1}
                  max={8}
                  step={1}
                  value={maxDepth}
                  disabled={Boolean(busy) || !recursive}
                  onChange={(event) => {
                    invalidate();
                    setMaxDepth(
                      Math.max(1, Math.min(8, Math.floor(Number(event.target.value) || 1))),
                    );
                  }}
                />
              </Field>
              {busy === 'scan' ? (
                <button type="button" className="button outline" onClick={invalidate}>
                  停止等待
                </button>
              ) : (
                <button
                  type="button"
                  className="button outline"
                  disabled={!nodeId || Boolean(pathError) || Boolean(busy)}
                  onClick={() => void scanDirectory()}
                >
                  <FolderSearch size={16} />
                  {scan ? '重新扫描' : '扫描数据集'}
                </button>
              )}
            </div>
          </section>
          {busy === 'scan' && (
            <p className="batch-dataset-scanning" role="status">
              <LoaderCircle size={16} className="spin" />
              正在读取目录和数据集元信息…关闭窗口会停止等待，服务端只读扫描会自行结束。
            </p>
          )}
          {scan?.state === 'READY' && (
            <>
              <section className="batch-dataset-preview" aria-label="扫描结果">
                <div className="batch-dataset-preview-heading">
                  <h3>扫描结果 · {rows.length} 个数据集</h3>
                  <span>
                    已勾选 {selected.length} 个 · 遍历 {scan.scanned_directories} 个目录
                  </span>
                </div>
                {scan.truncated && (
                  <p className="batch-dataset-warning" role="status">
                    扫描达到数量、深度或读取预算限制，本次结果可能不完整；请缩小目录或调整深度后继续扫描。
                  </p>
                )}
                {!!scan.warnings.length && (
                  <details className="batch-dataset-warnings">
                    <summary>扫描提示（{scan.warnings.length}）</summary>
                    <ul>
                      {scan.warnings.map((warning, index) => (
                        <li key={index}>{warning}</li>
                      ))}
                    </ul>
                  </details>
                )}
                {rows.length ? (
                  <div className="table-scroll">
                    <table className="batch-dataset-table" aria-label="批量数据集预览">
                      <thead>
                        <tr>
                          <th>
                            <input
                              type="checkbox"
                              aria-label="选择全部可登记数据集"
                              disabled={Boolean(busy) || !eligible.length}
                              checked={
                                Boolean(eligible.length) && eligible.every((row) => row.selected)
                              }
                              onChange={(event) =>
                                setRows((items) =>
                                  items.map((row) => ({
                                    ...row,
                                    selected:
                                      !row.entry.error &&
                                      !row.existingAssetId &&
                                      event.target.checked,
                                  })),
                                )
                              }
                            />
                          </th>
                          <th>资产名称</th>
                          <th>数据集路径</th>
                          <th>元信息</th>
                          <th>状态</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row, index) => (
                          <Fragment key={row.entry.path}>
                            <tr className="batch-dataset-info-row">
                              <td>
                                <input
                                  type="checkbox"
                                  aria-label={`选择数据集 ${row.entry.relative_path || row.entry.name}`}
                                  checked={row.selected}
                                  disabled={
                                    Boolean(busy) ||
                                    Boolean(row.entry.error) ||
                                    Boolean(row.existingAssetId)
                                  }
                                  onChange={(event) =>
                                    setRows((items) =>
                                      items.map((item, position) =>
                                        position === index
                                          ? { ...item, selected: event.target.checked }
                                          : item,
                                      ),
                                    )
                                  }
                                />
                              </td>
                              <td>
                                <input
                                  aria-label={`数据集名称 ${index + 1}`}
                                  maxLength={200}
                                  value={row.name}
                                  disabled={
                                    Boolean(busy) ||
                                    Boolean(row.entry.error) ||
                                    Boolean(row.existingAssetId)
                                  }
                                  onChange={(event) =>
                                    setRows((items) =>
                                      items.map((item, position) =>
                                        position === index
                                          ? { ...item, name: event.target.value }
                                          : item,
                                      ),
                                    )
                                  }
                                />
                              </td>
                              <td>
                                <code>{row.entry.path}</code>
                              </td>
                              <td>
                                <span>{row.entry.summary.robot_type || '机器人待确认'}</span>
                                <span className="subtext">
                                  {row.entry.summary.total_episodes === undefined
                                    ? ''
                                    : `${row.entry.summary.total_episodes} 条轨迹`}
                                  {row.entry.summary.fps === undefined
                                    ? ''
                                    : ` · ${row.entry.summary.fps} Hz`}
                                </span>
                                {row.entry.summary.task_description && (
                                  <span
                                    className="subtext"
                                    title={row.entry.summary.task_description}
                                  >
                                    {row.entry.summary.task_description}
                                  </span>
                                )}
                              </td>
                              <td>
                                {row.entry.error ? (
                                  <span className="batch-dataset-error">
                                    {row.entry.error.message}
                                  </span>
                                ) : row.existingAssetId ? (
                                  <span className="batch-dataset-muted">
                                    已登记
                                    {row.entry.existing_state === 'ARCHIVED'
                                      ? '（已归档）'
                                      : row.entry.existing_state === 'UNAVAILABLE'
                                        ? '（不可用）'
                                        : ''}{' '}
                                    · 跳过
                                  </span>
                                ) : (
                                  '可登记'
                                )}
                              </td>
                            </tr>
                            {!row.entry.error && !row.existingAssetId && (
                              <tr className="batch-dataset-norm-row">
                                <td colSpan={5}>
                                  <div
                                    className={`batch-dataset-row-norm ${row.selected ? '' : 'unselected'}`}
                                    role="group"
                                    aria-label={`数据集 ${row.entry.relative_path || row.entry.name} 的 norm_stats`}
                                  >
                                    <div className="batch-dataset-row-norm-heading">
                                      <strong>归一化参数 · {row.name || row.entry.name}</strong>
                                      <span>
                                        {row.selected
                                          ? '仅绑定到当前数据集，可独立选择'
                                          : '未勾选：配置会保留，本次不登记或校验'}
                                      </span>
                                    </div>
                                    <DatasetNormField
                                      nodeId={nodeId}
                                      node={node}
                                      assets={assets.data || []}
                                      value={row.normId}
                                      onChange={(value) =>
                                        setRows((items) =>
                                          items.map((item, position) =>
                                            position === index ? { ...item, normId: value } : item,
                                          ),
                                        )
                                      }
                                      pathValue={row.normPath}
                                      onPathChange={(value) =>
                                        setRows((items) =>
                                          items.map((item, position) =>
                                            position === index
                                              ? { ...item, normPath: value }
                                              : item,
                                          ),
                                        )
                                      }
                                      disabled={Boolean(busy)}
                                    />
                                  </div>
                                </td>
                              </tr>
                            )}
                          </Fragment>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="field-hint">
                    未找到包含 meta/info.json 的 LeRobot 数据集，请检查父目录、扫描深度和访问权限。
                  </p>
                )}
              </section>
              {!!eligible.length && (
                <section className="batch-dataset-common" aria-label="公共登记配置">
                  <h3>公共登记配置 · 应用于本次勾选的 {selected.length} 个数据集</h3>
                  {selectedSignatures.size > 1 && (
                    <p className="batch-dataset-warning" role="status">
                      所选数据集存在不同字段结构，公共模板会统一应用；请核对或按格式分批登记。
                    </p>
                  )}
                  <div className="batch-dataset-common-basics">
                    <Field label="资产版本" required>
                      <input
                        aria-label="批量资产版本"
                        maxLength={200}
                        value={version}
                        disabled={Boolean(busy)}
                        onChange={(event) => setVersion(event.target.value)}
                      />
                    </Field>
                    <div
                      className="batch-dataset-shared-norm"
                      role="group"
                      aria-label="批量应用归一化参数"
                    >
                      <strong>归一化参数快捷填充</strong>
                      <DatasetNormField
                        nodeId={nodeId}
                        node={node}
                        assets={assets.data || []}
                        value={normId}
                        onChange={setNormId}
                        pathValue={normPath}
                        onPathChange={setNormPath}
                        disabled={Boolean(busy)}
                      />
                      <button
                        type="button"
                        className="button outline small"
                        disabled={Boolean(busy) || !selected.length || Boolean(sharedNormError)}
                        onClick={() => setRows((items) => applyBatchNorm(items, normId, normPath))}
                      >
                        应用到所选数据集
                      </button>
                      <p className="field-hint">
                        仅点击按钮后填入当前勾选项；之后仍可逐项修改。选择“不绑定”并应用可清空所选项。
                      </p>
                    </div>
                  </div>
                  <p className="field-hint">
                    每个数据集默认不绑定
                    norm_stats，请在上方逐项选择。只有共用同一份归一化参数时才使用快捷填充；选择公共值本身不会改动任何数据集。
                  </p>
                  <DatasetFormatTemplates
                    value={profile}
                    disabled={Boolean(busy)}
                    onApply={(value) => {
                      setProfile(value);
                      setMetadataOnly(false);
                    }}
                  />
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={metadataOnly}
                      disabled={Boolean(busy)}
                      onChange={(event) => setMetadataOnly(event.target.checked)}
                    />
                    仅登记路径，稍后补录数据集特征
                  </label>
                  {!metadataOnly && (
                    <details className="batch-dataset-profile-review">
                      <summary>
                        检查或编辑公共数据集特征 · {profile.robot_type || '机器人待填写'} ·{' '}
                        {profile.cameras.length} 个视觉输入 ·{' '}
                        {profile.field_annotations?.length || 0} 个字段标签
                      </summary>
                      <DatasetProfileForm
                        compact
                        value={profile}
                        onChange={setProfile}
                        disabled={Boolean(busy)}
                      />
                    </details>
                  )}
                  {registrationError && (
                    <p className="batch-dataset-warning" role="status">
                      {registrationError}
                    </p>
                  )}
                  {!metadataOnly && (
                    <p className="field-hint">
                      格式模板的字段标签和任务描述会复制到每个选中数据集；登记不会修改模板或来源数据集。
                    </p>
                  )}
                </section>
              )}
            </>
          )}
        </>
      )}
    </Modal>
  );
}
