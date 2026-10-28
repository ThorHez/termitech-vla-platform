import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { LoaderCircle, RefreshCw } from 'lucide-react';
import { api, ApiError } from './api';
import type { DatasetProfile } from './DatasetProfileForm';
import type { Node } from './types';
import {
  canonicalDatasetInspectionPath,
  clearAutomaticInspectionDefaults,
  inspectionMatches,
  inspectionScope,
  mergeDatasetInspection,
  recordInspectionEdits,
  type DatasetInspectionDTO,
  type DatasetInspectionQuery,
  type InspectionDraftState,
} from './datasetInspectionHelpers';
import './dataset-inspection.css';

export function useDatasetInspectionDraft(
  value: DatasetProfile,
  onChange: (value: DatasetProfile) => void,
  registered = false,
) {
  const state = useRef<InspectionDraftState | null>(null);
  if (!state.current)
    state.current = {
      initial: structuredClone(value),
      registered,
      touched: new Set(),
      automatic: new Map(),
    };
  const latest = useRef({ value, onChange });
  latest.current = { value, onChange };
  const manualChange = useCallback((next: DatasetProfile) => {
    recordInspectionEdits(state.current!, latest.current.value, next);
    latest.current.value = next;
    latest.current.onChange(next);
  }, []);
  const touch = useCallback((path: string) => {
    state.current!.touched.add(path);
  }, []);
  const apply = useCallback((inspection: DatasetInspectionDTO) => {
    const merged = mergeDatasetInspection(latest.current.value, inspection, state.current!);
    if (JSON.stringify(merged.profile) !== JSON.stringify(latest.current.value)) {
      latest.current.value = merged.profile;
      latest.current.onChange(merged.profile);
    }
    return merged;
  }, []);
  const currentScope = useRef<string | undefined>(undefined);
  const resetScope = useCallback((scope: string) => {
    if (currentScope.current !== undefined && currentScope.current !== scope) {
      const next = clearAutomaticInspectionDefaults(latest.current.value, state.current!);
      if (JSON.stringify(next) !== JSON.stringify(latest.current.value)) {
        latest.current.value = next;
        latest.current.onChange(next);
      }
    }
    currentScope.current = scope;
  }, []);
  return { manualChange, touch, apply, resetScope };
}

function failure(error: unknown) {
  if (
    error instanceof ApiError &&
    ['NODE_OFFLINE', 'NODE_UNAVAILABLE', 'NODE_NOT_ONLINE'].includes(error.code)
  )
    return '机器当前离线，暂不能读取数据集字段。Agent 在线后可重试；仍可手动填写。';
  return error instanceof Error ? error.message : '数据集字段读取失败，请重试或继续手动填写。';
}
const wait = (signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const stop = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', stop);
      resolve();
    }, 900);
    signal.addEventListener('abort', stop, { once: true });
  });

function InspectionDetails({
  compact,
  warningCount,
  children,
}: {
  compact: boolean;
  warningCount: number;
  children: ReactNode;
}) {
  if (!compact) return <>{children}</>;
  return (
    <details className="dataset-inspection-compact-details">
      <summary>
        读取详情
        {warningCount > 0 && (
          <span className="dataset-inspection-warning-count"> · 需核对 {warningCount} 项</span>
        )}
      </summary>
      {children}
    </details>
  );
}

export function DatasetInspection({
  node,
  path,
  value,
  draft,
  disabled = false,
  compact = false,
}: {
  node?: Node;
  path: string;
  value: DatasetProfile;
  draft: ReturnType<typeof useDatasetInspectionDraft>;
  disabled?: boolean;
  compact?: boolean;
}) {
  const [revision, setRevision] = useState(0);
  const [view, setView] = useState<{
    scope: string;
    loading?: boolean;
    inspection?: DatasetInspectionDTO;
    error?: string;
    applied?: string[];
    kept?: string[];
  }>();
  const scope = inspectionScope(node?.id, path),
    canonicalPath = canonicalDatasetInspectionPath(path);
  const online = node?.status === 'ONLINE';
  const ready = !!node && online && !!canonicalPath && !disabled;
  const latest = useRef({ scope, ready, draft });
  latest.current = { scope, ready, draft };
  const generation = useRef(0);
  const current = view?.scope === scope ? view : undefined;
  useEffect(() => {
    latest.current.draft.resetScope(scope);
    const serial = ++generation.current,
      abort = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (ready && node) {
      setView({ scope, loading: true });
      timer = setTimeout(() => {
        const active = () =>
          !abort.signal.aborted &&
          serial === generation.current &&
          latest.current.scope === scope &&
          latest.current.ready;
        void (async () => {
          try {
            let result = await api<DatasetInspectionQuery>(
              `/nodes/${encodeURIComponent(node.id)}/dataset-inspections`,
              {
                method: 'POST',
                body: JSON.stringify({ path: canonicalPath }),
                signal: AbortSignal.any([abort.signal, AbortSignal.timeout(20000)]),
              },
            );
            if (!active()) return;
            if (!inspectionMatches(result, node.id, canonicalPath))
              throw new Error('读取结果与当前机器或目录不一致，请重试。');
            const queryId = result.id,
              deadline = Date.now() + 65000;
            while (result.status === 'PENDING' && active()) {
              if (Date.now() > deadline)
                throw new Error('读取数据集字段超时，请确认 Agent 在线后重试。');
              await wait(abort.signal);
              if (!active()) return;
              result = await api<DatasetInspectionQuery>(
                `/nodes/${encodeURIComponent(node.id)}/dataset-inspections/${encodeURIComponent(queryId)}`,
                { signal: AbortSignal.any([abort.signal, AbortSignal.timeout(20000)]) },
              );
              if (!active()) return;
              if (!inspectionMatches(result, node.id, canonicalPath, queryId))
                throw new Error('读取结果已失效，请按当前目录重新读取。');
            }
            if (!active()) return;
            if (result.status !== 'READY' || !result.inspection)
              throw new Error(result.error?.message || '未能读取数据集元信息，请重试。');
            const merged = latest.current.draft.apply(result.inspection);
            setView({
              scope,
              inspection: result.inspection,
              applied: merged.applied,
              kept: merged.kept,
            });
          } catch (error) {
            if (active()) setView({ scope, error: failure(error) });
          }
        })();
      }, 800);
    } else {
      setView((previous) => (previous?.scope === scope && previous.loading ? { scope } : previous));
    }
    return () => {
      if (timer) clearTimeout(timer);
      abort.abort();
    };
  }, [scope, online, disabled, revision]);
  // Enabling composition after inspection reuses only actual field candidates;
  // it does not enable the mode or complete unknown physical declarations.
  useEffect(() => {
    if (current?.inspection && ready && value.processing) {
      const merged = latest.current.draft.apply(current.inspection);
      setView((previous) =>
        previous?.scope === scope && previous.inspection === current.inspection
          ? { ...previous, applied: [...new Set([...(previous.applied || []), ...merged.applied])] }
          : previous,
      );
    }
  }, [!!value.processing, current?.inspection, ready]);
  return (
    <section
      className={`dataset-inspection${compact ? ' compact' : ''}`}
      aria-label="数据集字段读取"
    >
      <button
        type="button"
        className="button outline small"
        disabled={!ready || !!current?.loading}
        onClick={() => setRevision((previous) => previous + 1)}
      >
        {current?.loading ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}
        {current?.loading ? '正在读取数据集字段…' : '读取数据集字段，推断并填入默认值'}
      </button>
      <p className="field-hint">
        {compact
          ? '自动读取元信息；预填值可直接修改。'
          : '机器和绝对路径稳定后自动读取。仅读取数据集元信息，不启动训练；填入的是待确认草稿，请核对后使用现有保存按钮。'}
      </p>
      {!node || !canonicalPath ? (
        <p className="field-hint">
          {compact
            ? '先选择机器并填写数据集路径。'
            : '先选择机器并填写数据集绝对路径；也可以继续手动填写特征。'}
        </p>
      ) : !online ? (
        <p className="inline-note amber" role="status">
          机器当前离线，实际读取需要 Agent 在线；可继续手动填写，在线后重试。
        </p>
      ) : null}
      {current?.error && (
        <p className="recipe-detail-warning" role="status">
          {current.error}
        </p>
      )}
      {current?.inspection && (
        <div className="dataset-inspection-result" role="status">
          <strong>已读取 {current.inspection.features.length} 个真实字段；推断待核对</strong>
          <InspectionDetails compact={compact} warningCount={current.inspection.warnings.length}>
            <p className="field-hint">
              只补充未填写或未编辑的默认项，保留已登记值和本次手工修改。组合候选沿用草稿的归一化与视觉槽位，单位、坐标系和夹爪范围仍需明确填写。
            </p>
            <p className="field-hint">
              本次候选更新 {current.applied?.length || 0} 项；保留 {current.kept?.length || 0}{' '}
              项已有设置。信息摘要 {current.inspection.source.info_sha256.slice(0, 12)}。
            </p>
            {current.inspection.warnings.map((warning, index) => (
              <p className="recipe-detail-warning" key={index}>
                {warning}
              </p>
            ))}
            <details>
              <summary>查看来源与推断依据</summary>
              <ul>
                {current.inspection.evidence.map((entry, index) => (
                  <li key={index}>
                    <strong>
                      {entry.kind === 'metadata' ? '元信息' : '推断待核对'} · {entry.field}
                    </strong>
                    ：{entry.reason}
                    {entry.keys.length ? `（${entry.keys.join('、')}）` : ''}
                  </li>
                ))}
              </ul>
            </details>
            <details>
              <summary>查看实际字段与维度</summary>
              <div className="dataset-inspection-fields">
                <table>
                  <thead>
                    <tr>
                      <th>真实字段</th>
                      <th>类型与形状</th>
                      <th>维度名称</th>
                    </tr>
                  </thead>
                  <tbody>
                    {current.inspection.features.map((feature) => (
                      <tr key={feature.key}>
                        <td>
                          <code>{feature.key}</code>
                        </td>
                        <td>
                          {feature.dtype} [{feature.shape.join(', ')}]
                        </td>
                        <td>
                          {feature.names.slice(0, 16).join('、')}
                          {feature.names.length > 16 ? `… 共 ${feature.names.length} 项` : ''}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </InspectionDetails>
        </div>
      )}
    </section>
  );
}
