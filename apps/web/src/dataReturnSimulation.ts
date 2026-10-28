/** Local example data only. This module never controls a robot or submits work. */

export interface DataReturnContext {
  deploymentRecordId: string;
  name: string;
  machine: string;
  runtimeId?: string;
}

export type DataReturnMode = 'shadow' | 'dagger';
export type DataReturnSessionStatus =
  | 'draft'
  | 'running'
  | 'paused'
  | 'takeover'
  | 'between_episodes'
  | 'completed';
export type DataReturnReview = 'pending' | 'retained' | 'excluded';
export type DataReturnEpisodeOutcome = 'success' | 'failure' | 'unknown';

export interface DataReturnAnnotation {
  id: string;
  startFrame: number;
  endFrame: number;
  outcome: 'unknown' | 'success' | 'failure';
  issues: (
    | 'grasp_failure'
    | 'pose_error'
    | 'collision_risk'
    | 'occlusion'
    | 'unstable_motion'
    | 'other'
  )[];
  intent:
    | 'none'
    | 'align_pose'
    | 'adjust_gripper'
    | 'regrasp'
    | 'recover_trajectory'
    | 'complete_task';
  note: string;
}

export type DataReturnAnnotationInput = Omit<DataReturnAnnotation, 'id'>;

export interface DataReturnAnnotationSnapshot
  extends Readonly<Omit<DataReturnAnnotation, 'issues'>> {
  readonly issues: readonly DataReturnAnnotation['issues'][number][];
}

export interface DataReturnSegment {
  id: string;
  startFrame: number;
  endFrame: number;
  startMs: number;
  endMs: number;
  source: 'policy' | 'human';
  expertValid: boolean;
  review: DataReturnReview;
  note: string;
  annotations: DataReturnAnnotation[];
}

export interface DataReturnEpisode {
  id: string;
  name: string;
  status: 'recording' | 'sealed';
  outcome: DataReturnEpisodeOutcome;
  startedAt?: number;
  finishedAt?: number;
  durationMs: number;
  frames: number;
  fps: number;
  segments: DataReturnSegment[];
}

export interface DataReturnSession {
  id: string;
  name: string;
  task: string;
  mode: DataReturnMode;
  status: DataReturnSessionStatus;
  createdAt: number;
  startedAt?: number;
  finishedAt?: number;
  lastUpdatedAt: number;
  durationMs: number;
  frames: number;
  takeovers: number;
  episodes: DataReturnEpisode[];
}

export interface DataReturnSegmentSnapshot
  extends Readonly<Omit<DataReturnSegment, 'annotations'>> {
  readonly annotations: readonly DataReturnAnnotationSnapshot[];
  readonly sessionId: string;
  readonly sessionName: string;
  readonly episodeId: string;
  readonly episodeName: string;
  readonly episodeOutcome: DataReturnEpisodeOutcome;
  readonly frameCount: number;
}

export interface DataReturnVersion {
  readonly id: string;
  readonly name: string;
  readonly createdAt: number;
  readonly frameCount: number;
  readonly sessionIds: readonly string[];
  readonly sourceSegmentIds: readonly string[];
  readonly segments: readonly DataReturnSegmentSnapshot[];
  readonly context: Readonly<DataReturnContext>;
  readonly simulationOnly: true;
  /** Selection identity for local idempotency; this is not a content SHA. */
  readonly contentKey: string;
}

export interface DataReturnSimulation {
  context: DataReturnContext;
  sessions: DataReturnSession[];
  versions: DataReturnVersion[];
  activeSessionId: string;
  lastPublishedVersionId: string;
  nextSequence: number;
  nextAnnotationSequence: number;
}

export type DataReturnAction =
  | { type: 'create-session'; mode: DataReturnMode; name?: string; task?: string; now: number }
  | { type: 'start' | 'pause' | 'takeover' | 'resume' | 'finish'; sessionId: string; now: number }
  | {
      type: 'finish-episode';
      sessionId: string;
      episodeId: string;
      outcome: DataReturnEpisodeOutcome;
      now: number;
    }
  | { type: 'next-episode'; sessionId: string; afterEpisodeId: string; now: number }
  | { type: 'tick'; now: number }
  | {
      type: 'review';
      sessionId: string;
      episodeId: string;
      segmentId: string;
      review: DataReturnReview;
      note?: string;
    }
  | {
      type: 'annotate';
      sessionId: string;
      episodeId: string;
      segmentId: string;
      annotation: DataReturnAnnotationInput;
      annotationId?: string;
    }
  | {
      type: 'remove-annotation';
      sessionId: string;
      episodeId: string;
      segmentId: string;
      annotationId: string;
    }
  | { type: 'publish'; segmentIds?: string[]; name?: string; now: number }
  | { type: 'reset'; now: number };

const FPS = 20;
const minute = 60_000;
const frameAt = (milliseconds: number) => Math.floor((milliseconds * FPS) / 1000);
const recording = (status: DataReturnSessionStatus) =>
  status === 'running' || status === 'takeover';
const occupiesSession = (status: DataReturnSessionStatus) =>
  recording(status) || status === 'paused' || status === 'between_episodes';

const annotationOutcomes = ['unknown', 'success', 'failure'];
const annotationIssues = [
  'grasp_failure',
  'pose_error',
  'collision_risk',
  'occlusion',
  'unstable_motion',
  'other',
];
const annotationIntents = [
  'none',
  'align_pose',
  'adjust_gripper',
  'regrasp',
  'recover_trajectory',
  'complete_task',
];

export function annotationValidationError(
  segment: DataReturnSegment,
  input: DataReturnAnnotationInput,
): string | undefined {
  if (!input || typeof input !== 'object') return '请填写标注内容。';
  if (
    !Number.isInteger(input.startFrame) ||
    !Number.isInteger(input.endFrame) ||
    input.startFrame < segment.startFrame ||
    input.endFrame > segment.endFrame ||
    input.endFrame <= input.startFrame
  )
    return '标注帧区间需为片段内非空的整数范围，结束帧不包含在区间内。';
  if (!annotationOutcomes.includes(input.outcome)) return '请选择有效的任务结果。';
  if (
    !Array.isArray(input.issues) ||
    Array.from(input.issues).some(
      (issue) => typeof issue !== 'string' || !annotationIssues.includes(issue),
    ) ||
    new Set(input.issues).size !== input.issues.length
  )
    return '请选择有效且不重复的问题类型。';
  if (!annotationIntents.includes(input.intent)) return '请选择有效的纠正意图。';
  if (typeof input.note !== 'string' || input.note.length > 240)
    return '标注说明需为不超过 240 字的文本。';
  if (
    input.outcome === 'unknown' &&
    !input.issues.length &&
    input.intent === 'none' &&
    !input.note.trim()
  )
    return '请至少填写任务结果、问题类型、纠正意图或标注说明中的一项。';
  return undefined;
}

function deriveSession(session: DataReturnSession): DataReturnSession {
  return {
    ...session,
    durationMs: session.episodes.reduce((sum, episode) => sum + episode.durationMs, 0),
    frames: session.episodes.reduce((sum, episode) => sum + episode.frames, 0),
    takeovers:
      session.mode === 'shadow'
        ? 0
        : session.episodes.reduce(
            (sum, episode) =>
              sum +
              episode.segments.filter(
                (segment, index) =>
                  segment.source === 'human' && episode.segments[index - 1]?.source !== 'human',
              ).length,
            0,
          ),
  };
}

function segment(
  id: string,
  startMs: number,
  endMs: number,
  source: 'policy' | 'human',
  review: DataReturnReview = 'pending',
  expertValid = source === 'human',
  note = '',
): DataReturnSegment {
  return {
    id,
    startMs,
    endMs,
    startFrame: frameAt(startMs),
    endFrame: frameAt(endMs),
    source,
    expertValid,
    review,
    note,
    annotations: [],
  };
}

function episode(
  id: string,
  name: string,
  durationMs: number,
  segments: DataReturnSegment[],
  outcome: DataReturnEpisodeOutcome = 'unknown',
): DataReturnEpisode {
  return {
    id,
    name,
    status: 'sealed',
    outcome,
    durationMs,
    frames: frameAt(durationMs),
    fps: FPS,
    segments,
  };
}

function eligibleSegments(state: DataReturnSimulation, selected?: readonly string[]) {
  const ids = selected === undefined ? undefined : new Set(selected);
  return state.sessions.flatMap((session) =>
    session.episodes.flatMap((episode) =>
      episode.status !== 'sealed'
        ? []
        : episode.segments
            .filter(
              (item) =>
                item.source === 'human' &&
                item.expertValid &&
                item.review === 'retained' &&
                item.endFrame > item.startFrame &&
                (ids === undefined || ids.has(item.id)),
            )
            .map((item) => ({ session, episode, segment: item })),
    ),
  );
}

function versionSnapshot(
  state: DataReturnSimulation,
  selected: ReturnType<typeof eligibleSegments>,
  name: string,
  now: number,
): DataReturnVersion {
  const snapshots = selected.map(({ session, episode, segment }) =>
    Object.freeze({
      ...segment,
      annotations: Object.freeze(
        segment.annotations.map((annotation) =>
          Object.freeze({ ...annotation, issues: Object.freeze([...annotation.issues]) }),
        ),
      ),
      sessionId: session.id,
      sessionName: session.name,
      episodeId: episode.id,
      episodeName: episode.name,
      episodeOutcome: episode.outcome,
      frameCount: segment.endFrame - segment.startFrame,
    }),
  );
  const contentKey = JSON.stringify(
    snapshots
      .map((item) => [
        item.id,
        item.startFrame,
        item.endFrame,
        item.source,
        item.episodeOutcome,
        item.annotations
          .map((annotation) => [
            annotation.startFrame,
            annotation.endFrame,
            annotation.outcome,
            [...annotation.issues].sort(),
            annotation.intent,
            annotation.note,
          ])
          .sort((a, b) => {
            const left = JSON.stringify(a),
              right = JSON.stringify(b);
            return left < right ? -1 : left > right ? 1 : 0;
          }),
      ])
      .sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
  );
  return Object.freeze({
    id: `mock-version-${state.versions.length + 1}`,
    name,
    createdAt: now,
    frameCount: snapshots.reduce((sum, item) => sum + item.frameCount, 0),
    sessionIds: Object.freeze([...new Set(snapshots.map((item) => item.sessionId))]),
    sourceSegmentIds: Object.freeze(snapshots.map((item) => item.id)),
    segments: Object.freeze(snapshots),
    context: Object.freeze({ ...state.context }),
    simulationOnly: true,
    contentKey,
  });
}

export function createDataReturnSimulation(
  context: DataReturnContext,
  now = Date.now(),
): DataReturnSimulation {
  const first = deriveSession({
    id: 'mock-session-1',
    name: '折叠衣服 · 纠错采集',
    task: '折叠衣服',
    mode: 'dagger',
    status: 'completed',
    createdAt: now - 22 * minute,
    startedAt: now - 20 * minute - 70_000,
    finishedAt: now - 20 * minute,
    lastUpdatedAt: now - 20 * minute,
    durationMs: 0,
    frames: 0,
    takeovers: 0,
    episodes: [
      episode(
        'mock-episode-1',
        'Episode 001',
        40_000,
        [
          segment(
            'mock-segment-1',
            0,
            12_000,
            'policy',
            'excluded',
            false,
            '自主动作仅保留供回放，不作为专家监督。',
          ),
          segment(
            'mock-segment-2',
            12_000,
            24_000,
            'human',
            'retained',
            true,
            '人工纠正左腕角度，已审核保留。',
          ),
          segment('mock-segment-3', 24_000, 40_000, 'policy', 'excluded', false),
        ],
        'success',
      ),
      episode(
        'mock-episode-2',
        'Episode 002',
        30_000,
        [
          segment('mock-segment-4', 0, 10_000, 'policy', 'pending', false),
          segment(
            'mock-segment-5',
            10_000,
            20_000,
            'human',
            'pending',
            true,
            '夹取失败后人工恢复，待审核。',
          ),
          segment(
            'mock-segment-6',
            20_000,
            25_000,
            'human',
            'pending',
            false,
            '专家信号未对齐，不能发布为有效标签。',
          ),
          segment('mock-segment-7', 25_000, 30_000, 'policy', 'excluded', false),
        ],
        'failure',
      ),
    ],
  });
  const second = deriveSession({
    id: 'mock-session-2',
    name: '折叠衣服 · 影子采集',
    task: '折叠衣服',
    mode: 'shadow',
    status: 'completed',
    createdAt: now - 17 * minute,
    startedAt: now - 15 * minute - 45_000,
    finishedAt: now - 15 * minute,
    lastUpdatedAt: now - 15 * minute,
    durationMs: 0,
    frames: 0,
    takeovers: 0,
    episodes: [
      episode('mock-episode-3', 'Episode 003', 45_000, [
        segment(
          'mock-segment-8',
          0,
          45_000,
          'human',
          'excluded',
          false,
          '影子采集由人控执行，但未采集有效专家动作通道，不可作为监督。',
        ),
      ]),
    ],
  });
  const draft: DataReturnSession = {
    id: 'mock-session-3',
    name: '本地采集',
    task: '折叠衣服',
    mode: 'dagger',
    status: 'draft',
    createdAt: now,
    lastUpdatedAt: now,
    durationMs: 0,
    frames: 0,
    takeovers: 0,
    episodes: [],
  };
  const state: DataReturnSimulation = {
    context: { ...context },
    sessions: [first, second, draft],
    versions: [],
    activeSessionId: draft.id,
    lastPublishedVersionId: '',
    nextSequence: 4,
    nextAnnotationSequence: 2,
  };
  first.episodes[0].segments[1].annotations.push({
    id: 'mock-annotation-1',
    startFrame: 240,
    endFrame: 480,
    outcome: 'success',
    issues: ['pose_error'],
    intent: 'align_pose',
    note: '调整左腕姿态后完成对齐。',
  });
  const initial = versionSnapshot(state, eligibleSegments(state), '回流版本 v1', now - 10 * minute);
  return { ...state, versions: [initial], lastPublishedVersionId: initial.id };
}

function advanceSession(session: DataReturnSession, requestedNow: number): DataReturnSession {
  const now = Number.isFinite(requestedNow)
    ? Math.max(session.lastUpdatedAt, requestedNow)
    : session.lastUpdatedAt;
  if (now === session.lastUpdatedAt) return session;
  if (!recording(session.status)) return { ...session, lastUpdatedAt: now };
  const current = session.episodes.at(-1);
  const active = current?.segments.at(-1);
  if (!current || current.status !== 'recording' || !active) return session;
  const durationMs = current.durationMs + now - session.lastUpdatedAt;
  const next = {
    ...current,
    durationMs,
    frames: frameAt(durationMs),
    segments: [
      ...current.segments.slice(0, -1),
      { ...active, endMs: durationMs, endFrame: frameAt(durationMs) },
    ],
  };
  return deriveSession({
    ...session,
    lastUpdatedAt: now,
    episodes: [...session.episodes.slice(0, -1), next],
  });
}

function appendControlSegment(
  session: DataReturnSession,
  source: 'policy' | 'human',
): DataReturnSession {
  const current = session.episodes.at(-1);
  if (!current) return session;
  const next = {
    ...current,
    segments: [
      ...current.segments,
      segment(
        `${current.id}-segment-${current.segments.length + 1}`,
        current.durationMs,
        current.durationMs,
        source,
      ),
    ],
  };
  return deriveSession({ ...session, episodes: [...session.episodes.slice(0, -1), next] });
}

function recordingEpisode(session: DataReturnSession, now: number): DataReturnEpisode {
  const sequence = session.episodes.length + 1;
  const id = `${session.id}-episode-${sequence}`;
  return {
    id,
    name: `Episode ${String(sequence).padStart(3, '0')}`,
    status: 'recording',
    outcome: 'unknown',
    startedAt: now,
    durationMs: 0,
    frames: 0,
    fps: FPS,
    segments: [segment(`${id}-segment-1`, 0, 0, session.mode === 'shadow' ? 'human' : 'policy')],
  };
}

export function dataReturnReducer(
  state: DataReturnSimulation,
  action: DataReturnAction,
): DataReturnSimulation {
  if (action.type === 'reset') return createDataReturnSimulation(state.context, action.now);
  if (action.type === 'create-session') {
    const id = `mock-session-${state.nextSequence}`;
    const session: DataReturnSession = {
      id,
      name: action.name?.trim() || `采集会话 ${state.nextSequence}`,
      task: action.task?.trim() || '折叠衣服',
      mode: action.mode,
      status: 'draft',
      createdAt: action.now,
      lastUpdatedAt: action.now,
      durationMs: 0,
      frames: 0,
      takeovers: 0,
      episodes: [],
    };
    return {
      ...state,
      sessions: [...state.sessions, session],
      activeSessionId: id,
      nextSequence: state.nextSequence + 1,
    };
  }
  if (action.type === 'publish') {
    const selected = eligibleSegments(state, action.segmentIds);
    if (!selected.length) return state;
    const snapshot = versionSnapshot(
      state,
      selected,
      action.name?.trim() || `回流版本 v${state.versions.length + 1}`,
      action.now,
    );
    const existing = state.versions.find((version) => version.contentKey === snapshot.contentKey);
    if (existing)
      return state.lastPublishedVersionId === existing.id
        ? state
        : { ...state, lastPublishedVersionId: existing.id };
    return {
      ...state,
      versions: [...state.versions, snapshot],
      lastPublishedVersionId: snapshot.id,
    };
  }
  if (action.type === 'annotate' || action.type === 'remove-annotation') {
    const session = state.sessions.find((item) => item.id === action.sessionId);
    const episode = session?.episodes.find((item) => item.id === action.episodeId);
    const target = episode?.segments.find((item) => item.id === action.segmentId);
    if (!target || episode?.status !== 'sealed' || target.endFrame <= target.startFrame)
      return state;
    const existing =
      action.annotationId === undefined
        ? undefined
        : target.annotations.find((item) => item.id === action.annotationId);
    if (action.annotationId !== undefined && !existing) return state;
    let annotations: DataReturnAnnotation[];
    let nextAnnotationSequence = state.nextAnnotationSequence;
    if (action.type === 'remove-annotation') {
      if (!existing) return state;
      annotations = target.annotations.filter((item) => item.id !== existing.id);
    } else {
      if (annotationValidationError(target, action.annotation)) return state;
      const annotation: DataReturnAnnotation = {
        id: existing?.id || `mock-annotation-${state.nextAnnotationSequence}`,
        startFrame: action.annotation.startFrame,
        endFrame: action.annotation.endFrame,
        outcome: action.annotation.outcome,
        issues: [...action.annotation.issues].sort(),
        intent: action.annotation.intent,
        note: action.annotation.note.trim(),
      };
      if (existing && JSON.stringify(existing) === JSON.stringify(annotation)) return state;
      annotations = existing
        ? target.annotations.map((item) => (item.id === existing.id ? annotation : item))
        : [...target.annotations, annotation];
      if (!existing) nextAnnotationSequence += 1;
    }
    return {
      ...state,
      nextAnnotationSequence,
      sessions: state.sessions.map((item) =>
        item !== session
          ? item
          : {
              ...item,
              episodes: item.episodes.map((part) =>
                part !== episode
                  ? part
                  : {
                      ...part,
                      segments: part.segments.map((segment) =>
                        segment !== target ? segment : { ...segment, annotations },
                      ),
                    },
              ),
            },
      ),
    };
  }
  if (action.type === 'review') {
    let changed = false;
    const sessions = state.sessions.map((session) =>
      session.id !== action.sessionId
        ? session
        : {
            ...session,
            episodes: session.episodes.map((episode) =>
              episode.id !== action.episodeId || episode.status !== 'sealed'
                ? episode
                : {
                    ...episode,
                    segments: episode.segments.map((item) => {
                      if (item.id !== action.segmentId) return item;
                      changed = true;
                      return {
                        ...item,
                        review: action.review,
                        note:
                          action.note === undefined ? item.note : action.note.trim().slice(0, 2000),
                      };
                    }),
                  },
            ),
          },
    );
    return changed ? { ...state, sessions } : state;
  }
  if (action.type === 'tick') {
    let changed = false;
    const sessions = state.sessions.map((session) => {
      if (!recording(session.status)) return session;
      const next = advanceSession(session, action.now);
      changed ||= next !== session;
      return next;
    });
    return changed ? { ...state, sessions } : state;
  }
  const session = state.sessions.find((item) => item.id === action.sessionId);
  if (!session || session.status === 'completed') return state;
  const anotherSessionOccupiesControl = state.sessions.some(
    (item) => item.id !== session.id && occupiesSession(item.status),
  );
  if (action.type === 'start' && (session.status !== 'draft' || anotherSessionOccupiesControl))
    return state;
  if (action.type === 'pause' && !recording(session.status)) return state;
  if (action.type === 'takeover' && (session.mode !== 'dagger' || session.status !== 'running'))
    return state;
  if (action.type === 'resume' && session.status !== 'paused' && session.status !== 'takeover')
    return state;
  if (action.type === 'resume' && anotherSessionOccupiesControl) return state;
  if (action.type === 'finish-episode') {
    const current = session.episodes.at(-1);
    if (
      !['running', 'takeover', 'paused'].includes(session.status) ||
      !current ||
      current.id !== action.episodeId ||
      current.status !== 'recording' ||
      !['success', 'failure', 'unknown'].includes(action.outcome)
    )
      return state;
  }
  if (action.type === 'next-episode') {
    const previous = session.episodes.at(-1);
    if (
      session.status !== 'between_episodes' ||
      anotherSessionOccupiesControl ||
      !previous ||
      previous.id !== action.afterEpisodeId ||
      previous.status !== 'sealed'
    )
      return state;
  }
  if (action.type === 'finish' && session.status === 'draft') return state;
  let next = advanceSession(session, action.now);
  const now = Math.max(
    session.lastUpdatedAt,
    Number.isFinite(action.now) ? action.now : session.lastUpdatedAt,
  );
  if (action.type === 'start') {
    next = {
      ...next,
      status: 'running',
      startedAt: now,
      lastUpdatedAt: now,
      episodes: [recordingEpisode(session, now)],
    };
  } else if (action.type === 'finish-episode') {
    next = {
      ...next,
      status: 'between_episodes',
      episodes: next.episodes.map((episode) =>
        episode.id !== action.episodeId
          ? episode
          : {
              ...episode,
              status: 'sealed',
              outcome: action.outcome,
              finishedAt: now,
            },
      ),
    };
  } else if (action.type === 'next-episode') {
    next = {
      ...next,
      status: 'running',
      lastUpdatedAt: now,
      episodes: [...next.episodes, recordingEpisode(next, now)],
    };
  } else if (action.type === 'pause') next = { ...next, status: 'paused' };
  else if (action.type === 'takeover')
    next = appendControlSegment({ ...next, status: 'takeover' }, 'human');
  else if (action.type === 'resume')
    next = appendControlSegment(
      { ...next, status: 'running' },
      session.mode === 'shadow' ? 'human' : 'policy',
    );
  else if (action.type === 'finish')
    next = {
      ...next,
      status: 'completed',
      finishedAt: now,
      episodes: next.episodes.map((episode) =>
        episode.status === 'sealed'
          ? episode
          : {
              ...episode,
              status: 'sealed' as const,
              outcome: 'unknown' as const,
              finishedAt: now,
            },
      ),
    };
  return {
    ...state,
    activeSessionId: next.id,
    sessions: state.sessions.map((item) => (item.id === next.id ? deriveSession(next) : item)),
  };
}

export function dataReturnStats(state: DataReturnSimulation) {
  const episodes = state.sessions.flatMap((session) => session.episodes);
  const sealed = episodes.filter((episode) => episode.status === 'sealed');
  const segments = episodes.flatMap((episode) => episode.segments);
  const eligible = eligibleSegments(state);
  const published = new Set(state.versions.flatMap((version) => [...version.sourceSegmentIds]));
  return {
    sessionCount: state.sessions.length,
    activeSessionCount: state.sessions.filter((session) => occupiesSession(session.status)).length,
    recordingSessionCount: state.sessions.filter((session) => recording(session.status)).length,
    completedSessionCount: state.sessions.filter((session) => session.status === 'completed')
      .length,
    episodeCount: episodes.length,
    sealedEpisodeCount: sealed.length,
    segmentCount: segments.length,
    frameCount: episodes.reduce((sum, episode) => sum + episode.frames, 0),
    durationMs: episodes.reduce((sum, episode) => sum + episode.durationMs, 0),
    takeovers: state.sessions.reduce((sum, session) => sum + session.takeovers, 0),
    expertFrameCount: segments
      .filter((item) => item.source === 'human' && item.expertValid)
      .reduce((sum, item) => sum + item.endFrame - item.startFrame, 0),
    approvedFrameCount: eligible.reduce(
      (sum, item) => sum + item.segment.endFrame - item.segment.startFrame,
      0,
    ),
    publishableSegmentCount: eligible.length,
    unpublishedSegmentCount: eligible.filter(({ segment }) => !published.has(segment.id)).length,
    pendingReviewCount: sealed
      .flatMap((episode) => episode.segments)
      .filter((item) => item.review === 'pending').length,
    pendingExpertReviewCount: sealed
      .flatMap((episode) => episode.segments)
      .filter((item) => item.review === 'pending' && item.source === 'human' && item.expertValid)
      .length,
    versionCount: state.versions.length,
  };
}
