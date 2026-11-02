import { useCallback, useEffect, useReducer, useState } from 'react';
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  FolderSync,
  Plus,
  Search,
  Server,
  Workflow,
} from 'lucide-react';
import { date } from './api';
import { useAuth } from './auth';
import { Badge, Empty, ErrorNotice, Loading, PageHeading, ProgressBar } from './components';
import { useData } from './data';
import { assetTransferLabels } from './assetTransfer';
import { RunDeleteAction } from './RunDeleteAction';
import { AssetTransferDeleteAction } from './AssetTransferDeleteAction';
import type { Node, Template } from './types';
import {
  backgroundTaskActive,
  backgroundTaskPath,
  backgroundTaskProgress,
} from './backgroundTaskEntries';
import type { BackgroundTask, BackgroundTaskType } from './backgroundTaskEntries';
import {
  BACKGROUND_TASK_PAGE_SIZE,
  backgroundTaskPagePath,
  backgroundTaskPageRange,
  backgroundTaskQueryReducer,
  initialBackgroundTaskQuery,
} from './backgroundTaskPagination';
import type { BackgroundTaskPage, BackgroundTaskQuery } from './backgroundTaskPagination';
import { useBackgroundTaskPage } from './useBackgroundTaskPage';
import './background-tasks.css';

export function BackgroundTaskTable({
  tasks,
  nodes = [],
  templates = [],
  onRunDeleted,
  onTransferDeleted,
}: {
  tasks: BackgroundTask[];
  nodes?: Node[];
  templates?: Template[];
  onRunDeleted?: () => void;
  onTransferDeleted?: () => void;
}) {
  const machineName = (id: string) => nodes.find((node) => node.id === id)?.name || id.slice(0, 10);
  return (
    <div className="table-scroll background-task-table">
      <table>
        <thead>
          <tr>
            <th>任务名称</th>
            <th>任务类型</th>
            <th>状态</th>
            <th>进度</th>
            <th>机器</th>
            <th>创建时间</th>
            <th aria-label="任务操作" />
          </tr>
        </thead>
        <tbody>
          {tasks.map((task) => {
            const item = task.item;
            const progress = backgroundTaskProgress(task);
            const path = backgroundTaskPath(task);
            const sync = task.kind === 'data-sync';
            return (
              <tr key={task.key}>
                <td>
                  <a className="row-link" href={`#${path}`}>
                    <span className={`row-icon ${backgroundTaskActive(task) ? 'teal' : ''}`}>
                      {sync ? <FolderSync size={16} /> : <Workflow size={16} />}
                    </span>
                    <div>
                      <strong>{item.name}</strong>
                      <span className="subtext mono">{item.id.slice(0, 12)}</span>
                      {task.kind === 'training' && (
                        <span className="subtext">
                          {templates.find((template) => template.id === task.item.template_id)
                            ?.name || task.item.template_id}
                        </span>
                      )}
                    </div>
                  </a>
                </td>
                <td>
                  <span className={`tag background-task-type ${sync ? 'sync' : ''}`}>
                    {sync ? '数据同步' : '模型训练'}
                  </span>
                </td>
                <td>
                  <Badge
                    state={
                      sync && backgroundTaskActive(task)
                        ? 'STARTING'
                        : item.state === 'INTERRUPTED'
                          ? 'FAILED'
                          : item.state
                    }
                    label={sync ? assetTransferLabels[item.state] || item.state : undefined}
                  />
                </td>
                <td>
                  <div className="background-task-progress">
                    <span>{progress.label}</span>
                    {progress.preparing ? (
                      <div
                        className="progress background-task-indeterminate"
                        role="progressbar"
                        aria-label={progress.label}
                      >
                        <span />
                      </div>
                    ) : progress.percent !== null ? (
                      <ProgressBar value={progress.percent} tone={sync ? 'blue' : 'teal'} />
                    ) : null}
                    {progress.detail && <small>{progress.detail}</small>}
                  </div>
                </td>
                <td>
                  {task.kind === 'training' ? (
                    <a
                      className="node-link"
                      href={`#/nodes/${encodeURIComponent(task.item.node_id)}`}
                    >
                      <Server size={13} />
                      {machineName(task.item.node_id)}
                    </a>
                  ) : (
                    <div className="background-task-machines">
                      <a
                        className="node-link"
                        href={`#/nodes/${encodeURIComponent(task.item.source_node_id)}`}
                      >
                        {machineName(task.item.source_node_id)}
                      </a>
                      <ArrowRight size={12} />
                      <a
                        className="node-link"
                        href={`#/nodes/${encodeURIComponent(task.item.target_node_id)}`}
                      >
                        {machineName(task.item.target_node_id)}
                      </a>
                    </div>
                  )}
                </td>
                <td className="muted nowrap">{date(item.created_at)}</td>
                <td className="nowrap">
                  <a
                    className="icon-button"
                    href={`#${path}`}
                    aria-label={`查看 ${sync ? '数据同步' : '模型训练'} ${item.name}`}
                  >
                    <ChevronRight size={16} />
                  </a>
                  {task.kind === 'training' && (
                    <RunDeleteAction run={task.item} compact onDeleted={onRunDeleted} />
                  )}
                  {task.kind === 'data-sync' && (
                    <AssetTransferDeleteAction
                      item={task.item}
                      compact
                      onDeleted={onTransferDeleted}
                    />
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function BackgroundTaskErrors({
  trainingError,
  transferError,
  refreshTraining,
  refreshTransfers,
}: {
  trainingError?: Error;
  transferError?: Error;
  refreshTraining: () => void;
  refreshTransfers: () => void;
}) {
  return (
    <>
      <ErrorNotice error={trainingError} retry={refreshTraining} title="训练任务读取失败" />
      <ErrorNotice error={transferError} retry={refreshTransfers} title="数据同步任务读取失败" />
    </>
  );
}

export function BackgroundTasksPage({ createRun }: { createRun: () => void }) {
  const { can } = useAuth();
  const { enabled } = useData();
  const [request, dispatch] = useReducer(backgroundTaskQueryReducer, initialBackgroundTaskQuery);
  const [totalAll, setTotalAll] = useState<number>();
  const receivePage = useCallback((query: BackgroundTaskQuery, response: BackgroundTaskPage) => {
    setTotalAll(response.total_all);
    dispatch({ kind: 'receipt', request: query, page: response.page });
  }, []);
  return (
    <>
      <PageHeading
        eyebrow="BACKGROUND TASKS"
        title="训练任务"
        description="统一跟踪模型训练和数据同步；离开页面后，已提交任务继续在后台执行。"
        actions={
          can('training.write') && (
            <button className="button primary" disabled={!enabled} onClick={createRun}>
              <Plus size={17} />
              新建训练
            </button>
          )
        }
      />
      <div className="toolbar wrap background-task-toolbar">
        <div className="segmented">
          {(
            [
              ['ALL', '全部任务'],
              ['ACTIVE', '进行中'],
              ['SUCCEEDED', '已完成'],
              ['FAILED', '失败 / 中断'],
              ['CANCELLED', '已取消'],
            ] as const
          ).map(([key, label]) => (
            <button
              className={request.status === key ? 'selected' : ''}
              key={key}
              onClick={() => dispatch({ kind: 'status', value: key })}
            >
              {label}
              {key === 'ALL' && totalAll !== undefined && <b>{totalAll}</b>}
            </button>
          ))}
        </div>
        <div className="background-task-search">
          <label className="background-task-type-filter">
            任务类型
            <select
              aria-label="筛选任务类型"
              value={request.type}
              onChange={(event) =>
                dispatch({ kind: 'type', value: event.target.value as BackgroundTaskType })
              }
            >
              <option value="ALL">全部类型</option>
              <option value="training">模型训练</option>
              <option value="data-sync">数据同步</option>
            </select>
          </label>
          <div className="search-field">
            <Search size={16} />
            <input
              aria-label="搜索任务"
              placeholder="搜索任务名称或机器"
              maxLength={200}
              value={request.query}
              onChange={(event) => dispatch({ kind: 'query', value: event.target.value })}
            />
          </div>
        </div>
      </div>
      <BackgroundTaskResults
        key={backgroundTaskPagePath(request)}
        request={request}
        receivePage={receivePage}
        changePage={(page) => dispatch({ kind: 'page', page })}
      />
      <p className="background-task-history-note">
        每页固定显示 {BACKGROUND_TASK_PAGE_SIZE} 条，按创建时间从新到旧排列；每 3
        秒更新当前页，可翻页查看历史任务。
      </p>
    </>
  );
}

function BackgroundTaskResults({
  request,
  receivePage,
  changePage,
}: {
  request: BackgroundTaskQuery;
  receivePage: (query: BackgroundTaskQuery, response: BackgroundTaskPage) => void;
  changePage: (page: number) => void;
}) {
  const { nodes, templates, enabled, refresh } = useData();
  const result = useBackgroundTaskPage(request, enabled, nodes.data || []);
  useEffect(() => {
    if (result.data) receivePage(request, result.data);
  }, [result.data, request, receivePage]);
  const filtered = Boolean(request.query || request.type !== 'ALL' || request.status !== 'ALL');
  const deleted = () => {
    result.refresh();
    refresh();
  };
  return (
    <>
      <ErrorNotice error={result.error} retry={result.refresh} title="后台任务读取失败" />
      {result.data?.history_limited && (
        <p className="background-task-compatibility-note" role="status">
          当前服务暂未支持全量历史分页，显示已加载的 {result.data.total_all} 条记录（训练最多 500
          条，同步最多 200 条）。
        </p>
      )}
      <div className="panel">
        {enabled && result.loading && !result.data ? (
          <Loading label="正在读取后台任务…" />
        ) : result.data?.items.length ? (
          <BackgroundTaskTable
            tasks={result.data.items}
            nodes={nodes.data || []}
            templates={templates.data || []}
            onRunDeleted={deleted}
            onTransferDeleted={deleted}
          />
        ) : !result.error ? (
          <Empty
            icon={Workflow}
            title={filtered ? '没有匹配的任务' : '暂无任务'}
            description={
              filtered
                ? '调整任务类型、状态或搜索条件，查看其他任务。'
                : '创建模型训练，或从“数据与模型”将资产同步到其他机器。'
            }
          />
        ) : null}
      </div>
      <BackgroundTaskPagination
        result={result.data}
        loading={result.loading}
        changePage={changePage}
      />
    </>
  );
}

export function BackgroundTaskPagination({
  result,
  loading = false,
  changePage,
}: {
  result?: BackgroundTaskPage;
  loading?: boolean;
  changePage: (page: number) => void;
}) {
  const range = result ? backgroundTaskPageRange(result) : undefined;
  return (
    <nav className="background-task-pagination" aria-label="任务分页">
      <span aria-live="polite">
        {range && result
          ? `显示 ${range.start}–${range.end} 条，共 ${result.total} 条${result.history_limited ? '（已加载）' : ''}`
          : `每页 ${BACKGROUND_TASK_PAGE_SIZE} 条`}
      </span>
      <div className="background-task-page-controls">
        <button
          type="button"
          className="button outline small"
          disabled={!result || loading || result.page <= 1}
          onClick={() => result && changePage(result.page - 1)}
        >
          <ChevronLeft size={14} />
          上一页
        </button>
        <span>
          第 {result?.page || 1} / {result?.total_pages || 1} 页
        </span>
        <button
          type="button"
          className="button outline small"
          disabled={!result || loading || result.page >= result.total_pages}
          onClick={() => result && changePage(result.page + 1)}
        >
          下一页
          <ChevronRight size={14} />
        </button>
      </div>
    </nav>
  );
}
