import { useMemo, useState } from 'react';
import { ArrowRight, ChevronRight, FolderSync, Plus, Search, Server, Workflow } from 'lucide-react';
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
  buildBackgroundTasks,
  filterBackgroundTasks,
} from './backgroundTaskEntries';
import type { BackgroundTask, BackgroundTaskStatus, BackgroundTaskType } from './backgroundTaskEntries';
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
  const { runs, nodes, templates, enabled, assetTransfers: transfers, refresh } = useData();
  const [filter, setFilter] = useState<BackgroundTaskStatus>('ALL');
  const [type, setType] = useState<BackgroundTaskType>('ALL');
  const [query, setQuery] = useState('');
  const tasks = useMemo(
    () => buildBackgroundTasks(runs.data || [], transfers.data || []),
    [runs.data, transfers.data],
  );
  const filtered = useMemo(
    () => filterBackgroundTasks(tasks, { type, status: filter, query, nodes: nodes.data || [] }),
    [tasks, type, filter, query, nodes.data],
  );
  const loadingEmpty =
    enabled &&
    !tasks.length &&
    ((runs.loading && !runs.data) || (transfers.loading && !transfers.data));
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
              className={filter === key ? 'selected' : ''}
              key={key}
              onClick={() => setFilter(key)}
            >
              {label}
              {key === 'ALL' && <b>{tasks.length}</b>}
            </button>
          ))}
        </div>
        <div className="background-task-search">
          <label className="background-task-type-filter">
            任务类型
            <select
              aria-label="筛选任务类型"
              value={type}
              onChange={(event) => setType(event.target.value as BackgroundTaskType)}
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
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
        </div>
      </div>
      <BackgroundTaskErrors
        trainingError={runs.error}
        transferError={transfers.error}
        refreshTraining={runs.refresh}
        refreshTransfers={transfers.refresh}
      />
      <div className="panel">
        {loadingEmpty ? (
          <Loading label="正在读取后台任务…" />
        ) : filtered.length ? (
          <BackgroundTaskTable
            tasks={filtered}
            nodes={nodes.data || []}
            templates={templates.data || []}
            onRunDeleted={refresh}
            onTransferDeleted={refresh}
          />
        ) : (
          <Empty
            icon={Workflow}
            title={query || type !== 'ALL' || filter !== 'ALL' ? '没有匹配的任务' : '暂无任务'}
            description={
              query || type !== 'ALL' || filter !== 'ALL'
                ? '调整任务类型、状态或搜索条件，查看其他任务。'
                : '创建模型训练，或从“数据与模型”将资产同步到其他机器。'
            }
          />
        )}
      </div>
      <p className="background-task-history-note">
        同步记录每 3 秒更新，显示最近 200 条；点击任务查看进度、日志与操作。
      </p>
    </>
  );
}
