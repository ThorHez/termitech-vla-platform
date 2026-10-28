import {
  CheckCircle2,
  ChevronDown,
  Circle,
  Clapperboard,
  Clock3,
  Pause,
  Play,
  Square,
  XCircle,
} from 'lucide-react';
import type { DataReturnEpisodeOutcome, DataReturnSession } from './dataReturnSimulation';
import './data-return-episodes.css';

export interface DataReturnEpisodeControlProps {
  session: DataReturnSession;
  selectedEpisodeId?: string;
  nextBlockedReason?: string;
  onFinishEpisode(episodeId: string, outcome: DataReturnEpisodeOutcome): void;
  onStartNext(afterEpisodeId: string): void;
  onSelectEpisode(episodeId: string): void;
}

const outcomes: Record<DataReturnEpisodeOutcome, string> = {
  unknown: '未判定',
  success: '成功',
  failure: '失败',
};
const duration = (milliseconds: number) => {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
};

/** A local operator explicitly ends each Episode, then starts the next one. */
export default function DataReturnEpisodeControl({
  session,
  selectedEpisodeId,
  nextBlockedReason,
  onFinishEpisode,
  onStartNext,
  onSelectEpisode,
}: DataReturnEpisodeControlProps) {
  const latest = session.episodes.at(-1);
  if (session.status === 'draft' || !latest) return null;

  const active = ['running', 'takeover', 'paused'].includes(session.status);
  const canFinish = latest.status === 'recording' && active;
  const between = session.status === 'between_episodes' && latest.status === 'sealed';
  const paused = latest.status === 'recording' && session.status === 'paused';
  const completedCount = session.episodes.filter((episode) => episode.status === 'sealed').length;
  const nextName = `Episode ${String(session.episodes.length + 1).padStart(3, '0')}`;
  const viewingHistory =
    selectedEpisodeId && selectedEpisodeId !== latest.id
      ? session.episodes.find((episode) => episode.id === selectedEpisodeId)
      : undefined;

  return (
    <section
      className="dr-episode-control"
      aria-label="Episode 采集控制"
      data-testid="data-return-episode-control"
    >
      <div className="dre-heading">
        <div className="dre-title">
          <span className="dre-heading-icon">
            <Clapperboard size={15} />
          </span>
          <div>
            <span className="dre-eyebrow">本会话最新 Episode</span>
            <h4>{latest.name}</h4>
          </div>
          <span
            className={`dre-status ${latest.status === 'sealed' ? 'sealed' : paused ? 'paused' : 'recording'}`}
          >
            {latest.status === 'sealed' ? (
              <CheckCircle2 size={11} />
            ) : paused ? (
              <Pause size={10} />
            ) : (
              <Circle size={8} />
            )}
            {latest.status === 'sealed' ? '已结束' : paused ? '已暂停' : '录制中'}
          </span>
          {latest.status === 'sealed' && (
            <span className={`dre-outcome ${latest.outcome}`}>{outcomes[latest.outcome]}</span>
          )}
        </div>
        <div className="dre-stats">
          <span>
            <Clock3 size={11} />
            <b>{duration(latest.durationMs)}</b>
          </span>
          <span>
            <b>{latest.frames.toLocaleString()}</b> 帧
          </span>
          <span>
            已结束 <b>{completedCount}</b> 条
          </span>
        </div>
      </div>

      {viewingHistory && (
        <div className="dre-history-notice">
          <p>
            画面正在回放 {viewingHistory.name}；{latest.name}
            {latest.status === 'recording' ? (paused ? '已暂停' : '仍在采集') : '已结束'}。
            {canFinish && '“结束本条”只操作当前采集的 Episode。'}
          </p>
          {latest.status === 'recording' && (
            <button className="dre-replay" type="button" onClick={() => onSelectEpisode(latest.id)}>
              <Play size={10} />
              查看当前采集
            </button>
          )}
        </div>
      )}

      {canFinish && (
        <div className="dre-completion">
          <div className="dre-actions" aria-label="结束当前 Episode 并记录结果">
            <button
              className="dre-button primary"
              type="button"
              onClick={() => onFinishEpisode(latest.id, 'success')}
            >
              <CheckCircle2 size={12} />
              成功，结束本条
            </button>
            <button
              className="dre-button failure"
              type="button"
              onClick={() => onFinishEpisode(latest.id, 'failure')}
            >
              <XCircle size={12} />
              失败，结束本条
            </button>
            <button
              className="dre-button"
              type="button"
              onClick={() => onFinishEpisode(latest.id, 'unknown')}
            >
              <Square size={11} />
              中止本条
            </button>
          </div>
          <p>由操作员判断任务目标是否完成，结束本条将暂停采集，会话仍保留。</p>
        </div>
      )}

      {between && (
        <div className="dre-next">
          <p role="status">
            <strong>本条已结束，采集已暂停。</strong>整理场景后手动开始下一条；场景不会自动复位。
          </p>
          <div className="dre-next-action">
            <button
              className="dre-button primary"
              type="button"
              disabled={!!nextBlockedReason}
              title={nextBlockedReason || `手动开始 ${nextName}`}
              onClick={() => {
                if (!nextBlockedReason) onStartNext(latest.id);
              }}
            >
              <Play size={12} />
              开始 {nextName}
            </button>
            {nextBlockedReason ? (
              <span className="dre-blocked" role="status">
                {nextBlockedReason}
              </span>
            ) : (
              <span>沿用当前会话的任务与控制模式</span>
            )}
          </div>
        </div>
      )}

      {session.status === 'completed' && (
        <p className="dre-session-ended">
          本会话已结束，可回放与审核各条 Episode。开始新的采集请新建采集会话。
        </p>
      )}

      <details className="dre-history" aria-label="本会话 Episodes">
        <summary>
          <span>
            本会话 Episodes <b>{session.episodes.length}</b>
          </span>
          <span>
            查看每条结果与回放
            <ChevronDown size={13} />
          </span>
        </summary>
        <ol>
          {[...session.episodes].reverse().map((episode) => {
            const sealed = episode.status === 'sealed';
            const isLatest = episode.id === latest.id;
            return (
              <li className={selectedEpisodeId === episode.id ? 'selected' : ''} key={episode.id}>
                <div className="dre-history-name">
                  <strong>{episode.name}</strong>
                  {isLatest && <small>最新</small>}
                  <span className={`dre-outcome ${sealed ? episode.outcome : 'recording'}`}>
                    {sealed ? outcomes[episode.outcome] : paused && isLatest ? '已暂停' : '录制中'}
                  </span>
                </div>
                <div className="dre-history-meta">
                  <span>{sealed ? '已结束' : '未结束'}</span>
                  <span>{duration(episode.durationMs)}</span>
                  <span>{episode.frames.toLocaleString()} 帧</span>
                </div>
                {sealed ? (
                  <button
                    className="dre-replay"
                    type="button"
                    aria-label={`回放 ${episode.name}`}
                    onClick={() => onSelectEpisode(episode.id)}
                  >
                    <Play size={10} />
                    回放
                  </button>
                ) : (
                  <span className="dre-history-live">
                    {paused && isLatest ? '已暂停' : '采集中'}
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </details>
    </section>
  );
}
