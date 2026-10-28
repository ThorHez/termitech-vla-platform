import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const {
    createDataReturnSimulation: create,
    dataReturnReducer: reduce,
    dataReturnStats,
  } = await server.ssrLoadModule('/src/dataReturnSimulation.ts');
  const {
    createRobotDataWorkflow,
    robotDataWorkflowReducer: command,
    workflowNextEpisodeReason,
    workflowStartReason,
    workflowResumeReason,
    workflowStatus,
    activeWorkflowSession,
  } = await server.ssrLoadModule('/src/robotDataWorkflowSimulation.ts');
  const now = 3_000_000;
  const context = {
    deploymentRecordId: 'record-episodes',
    name: '多轮采集示例',
    machine: '示例机器',
  };
  const initial = () => create(context, now);
  const session = (state, id = state.activeSessionId) =>
    state.sessions.find((item) => item.id === id);
  const latest = (state, id) => session(state, id).episodes.at(-1);
  const act = (state, type, offset, more = {}) =>
    reduce(state, { type, sessionId: state.activeSessionId, now: now + offset, ...more });
  const prepare = (state) => {
    state = command(state, { type: 'runtime', action: 'start_sdk', now: now + 10 });
    const generation = state.runtimeGeneration;
    state = command(state, { type: 'runtime', action: 'sdk_ready', generation, now: now + 1110 });
    state = command(state, { type: 'runtime', action: 'start_eval', now: now + 1120 });
    return command(state, { type: 'runtime', action: 'eval_ready', generation, now: now + 2820 });
  };
  const workflowSession = (state, id = state.runtimeSessionId) => session(state.data, id);
  const workflowLatest = (state, id) => workflowSession(state, id).episodes.at(-1);

  await test('one session collects sequential episodes with explicit results and excludes pause and scene-reset time', () => {
    let state = act(initial(), 'start', 0);
    const id = state.activeSessionId;
    const firstId = latest(state).id;
    state = act(state, 'takeover', 1000);
    state = act(state, 'pause', 2000);
    state = act(state, 'finish-episode', 5000, { episodeId: firstId, outcome: 'success' });
    const first = latest(state);
    assert.equal(session(state).status, 'between_episodes');
    assert.equal(first.status, 'sealed');
    assert.equal(first.outcome, 'success');
    assert.equal(first.durationMs, 2000);
    assert.equal(first.frames, 40);
    assert.equal(first.startedAt, now);
    assert.equal(first.finishedAt, now + 5000);
    assert.equal(dataReturnStats(state).activeSessionCount, 1);
    assert.equal(dataReturnStats(state).recordingSessionCount, 0);
    assert.equal(reduce(state, { type: 'tick', now: now + 9000 }), state);
    assert.equal(act(state, 'resume', 9000), state);
    assert.equal(act(state, 'takeover', 9000), state);
    state = act(state, 'next-episode', 10000, { afterEpisodeId: firstId });
    const second = latest(state);
    assert.equal(state.activeSessionId, id);
    assert.equal(session(state).episodes[0], first);
    assert.equal(second.name, 'Episode 002');
    assert.equal(second.id, `${id}-episode-2`);
    assert.equal(second.startedAt, now + 10000);
    assert.equal(second.durationMs, 0);
    assert.equal(second.frames, 0);
    assert.equal(second.segments[0].startFrame, 0);
    assert.equal(second.segments[0].source, 'policy');
    state = act(state, 'finish-episode', 12000, { episodeId: second.id, outcome: 'failure' });
    assert.equal(latest(state).outcome, 'failure');
    state = act(state, 'next-episode', 20000, { afterEpisodeId: second.id });
    state = act(state, 'finish', 21000);
    assert.equal(session(state).status, 'completed');
    assert.equal(session(state).episodes.length, 3);
    assert.deepEqual(
      session(state).episodes.map((item) => item.outcome),
      ['success', 'failure', 'unknown'],
    );
    assert.deepEqual(
      session(state).episodes.map((item) => item.frames),
      [40, 40, 20],
    );
    assert.equal(session(state).durationMs, 5000);
    assert.equal(session(state).frames, 100);
    assert.equal(session(state).startedAt, now);
    assert.equal(session(state).finishedAt, now + 21000);
  });

  await test('episode-bound commands reject duplicates, stale ids, invalid outcomes and wrong lifecycle phases', () => {
    let state = act(initial(), 'start', 0);
    const firstId = latest(state).id;
    for (const more of [
      { episodeId: 'stale', outcome: 'success' },
      { episodeId: firstId, outcome: 'automatic_success' },
    ])
      assert.equal(act(state, 'finish-episode', 1000, more), state);
    assert.equal(act(state, 'next-episode', 1000, { afterEpisodeId: firstId }), state);
    state = act(state, 'finish-episode', 1000, { episodeId: firstId, outcome: 'unknown' });
    assert.equal(latest(state).outcome, 'unknown');
    assert.equal(
      act(state, 'finish-episode', 2000, { episodeId: firstId, outcome: 'success' }),
      state,
    );
    assert.equal(act(state, 'next-episode', 2000, { afterEpisodeId: 'old-episode' }), state);
    state = act(state, 'next-episode', 2000, { afterEpisodeId: firstId });
    assert.equal(act(state, 'next-episode', 2500, { afterEpisodeId: firstId }), state);
    assert.equal(
      act(state, 'finish-episode', 2500, { episodeId: firstId, outcome: 'success' }),
      state,
    );
    assert.equal(session(state).episodes[0].outcome, 'unknown');
    state = act(state, 'finish-episode', 3000, { episodeId: latest(state).id, outcome: 'failure' });
    const episodes = session(state).episodes;
    state = act(state, 'finish', 8000);
    assert.equal(session(state).episodes.length, 2);
    assert.equal(session(state).episodes[0], episodes[0]);
    assert.equal(session(state).episodes[1], episodes[1]);
    assert.equal(act(state, 'next-episode', 9000, { afterEpisodeId: episodes[1].id }), state);
  });

  await test('between-episode sessions retain exclusive control and historical labels remain reviewable', () => {
    let state = act(initial(), 'start', 0);
    const firstSessionId = state.activeSessionId;
    state = act(state, 'takeover', 1000);
    state = act(state, 'finish-episode', 2000, { episodeId: latest(state).id, outcome: 'success' });
    const first = latest(state);
    const human = first.segments.at(-1);
    state = reduce(state, {
      type: 'annotate',
      sessionId: firstSessionId,
      episodeId: first.id,
      segmentId: human.id,
      annotation: {
        startFrame: human.startFrame,
        endFrame: human.endFrame,
        outcome: 'success',
        issues: [],
        intent: 'complete_task',
        note: '首轮完成。',
      },
    });
    state = reduce(state, {
      type: 'review',
      sessionId: firstSessionId,
      episodeId: first.id,
      segmentId: human.id,
      review: 'retained',
    });
    state = reduce(state, { type: 'publish', segmentIds: [human.id], now: now + 2100 });
    const version = state.versions.at(-1);
    assert.equal(version.segments[0].episodeOutcome, 'success');
    assert.equal(version.segments[0].annotations[0].intent, 'complete_task');
    state = reduce(state, { type: 'create-session', mode: 'dagger', now: now + 3000 });
    assert.equal(act(state, 'start', 4000), state);
    const firstWithLabels = session(state, firstSessionId).episodes[0];
    state = reduce(state, {
      type: 'next-episode',
      sessionId: firstSessionId,
      afterEpisodeId: first.id,
      now: now + 5000,
    });
    assert.equal(session(state, firstSessionId).episodes[0], firstWithLabels);
    assert.equal(session(state, firstSessionId).episodes[1].segments[0].annotations.length, 0);
    assert.equal(version.segments[0].episodeOutcome, 'success');
    assert.equal(version.segments[0].annotations[0].note, '首轮完成。');
  });

  await test('shadow mode begins every episode with human control and no inferred success or takeovers', () => {
    let state = reduce(initial(), { type: 'create-session', mode: 'shadow', now });
    state = act(state, 'start', 0);
    const firstId = latest(state).id;
    state = act(state, 'finish-episode', 1000, { episodeId: firstId, outcome: 'failure' });
    state = act(state, 'next-episode', 5000, { afterEpisodeId: firstId });
    assert.equal(latest(state).segments[0].source, 'human');
    assert.equal(latest(state).segments[0].expertValid, true);
    state = act(state, 'finish', 6000);
    assert.deepEqual(
      session(state).episodes.map((item) => item.outcome),
      ['failure', 'unknown'],
    );
    assert.equal(session(state).takeovers, 0);
    assert.equal(session(state).durationMs, 2000);
  });

  await test('workflow pauses at episode end and resumes the same prepared session without another SDK/Eval cycle', () => {
    let state = prepare(createRobotDataWorkflow(context, now));
    const sessionId = state.data.activeSessionId;
    state = command(state, { type: 'start', sessionId, now: now + 3000 });
    const generation = state.runtimeGeneration;
    const firstId = workflowLatest(state).id;
    state = command(state, {
      type: 'finish-episode',
      sessionId,
      episodeId: firstId,
      outcome: 'success',
      now: now + 4000,
    });
    assert.equal(workflowSession(state).status, 'between_episodes');
    assert.equal(state.robot.phase, 'paused');
    assert.equal(activeWorkflowSession(state).id, sessionId);
    assert.equal(workflowStatus(state).source, 'hold');
    assert.match(workflowStatus(state).label, /下一 Episode/);
    assert.equal(workflowNextEpisodeReason(state, sessionId, firstId), undefined);
    assert.match(workflowNextEpisodeReason(state, sessionId, 'stale'), /已更新/);
    assert.match(workflowResumeReason(state, sessionId), /不能恢复/);
    const frames = workflowSession(state).frames;
    const motion = state.robot.motionMs;
    state = command(state, { type: 'tick', now: now + 14000 });
    assert.equal(workflowSession(state).frames, frames);
    assert.equal(state.robot.motionMs, motion);
    state = command(state, { type: 'create-session', mode: 'dagger', now: now + 14010 });
    const otherId = state.data.activeSessionId;
    assert.match(workflowStartReason(state, otherId), /结束当前/);
    assert.equal(command(state, { type: 'start', sessionId: otherId, now: now + 14020 }), state);
    assert.equal(
      command(state, {
        type: 'next-episode',
        sessionId: otherId,
        afterEpisodeId: firstId,
        now: now + 14020,
      }),
      state,
    );
    assert.equal(command(state, { type: 'runtime', action: 'start_sdk', now: now + 14020 }), state);
    state = command(state, {
      type: 'next-episode',
      sessionId,
      afterEpisodeId: firstId,
      now: now + 15000,
    });
    assert.equal(state.runtimeGeneration, generation);
    assert.equal(state.runtimeSessionId, sessionId);
    assert.equal(state.robot.phase, 'running');
    assert.equal(workflowLatest(state).frames, 0);
    assert.equal(workflowLatest(state).name, 'Episode 002');
    state = command(state, { type: 'tick', now: now + 16000 });
    assert.equal(workflowLatest(state).frames, 20);
    assert.equal(state.robot.motionMs, motion + 1000);
    const mismatched = { ...state, runtimeSessionId: otherId };
    assert.equal(
      command(mismatched, {
        type: 'finish-episode',
        sessionId,
        episodeId: workflowLatest(state).id,
        outcome: 'failure',
        now: now + 17000,
      }),
      mismatched,
    );
  });

  await test('outages close at episode boundaries, never relabel sealed history, and restart gaps on the next disconnected episode', () => {
    let state = prepare(createRobotDataWorkflow(context, now));
    state = command(state, { type: 'create-session', mode: 'shadow', now: now + 2900 });
    const sessionId = state.data.activeSessionId;
    state = command(state, { type: 'start', sessionId, now: now + 3000 });
    const firstId = workflowLatest(state).id;
    state = command(state, { type: 'runtime', action: 'stream_disconnect', now: now + 3500 });
    state = command(state, {
      type: 'finish-episode',
      sessionId,
      episodeId: firstId,
      outcome: 'failure',
      now: now + 4500,
    });
    const first = workflowLatest(state);
    assert.equal(state.streamGaps.length, 1);
    assert.equal(state.streamGaps[0].episodeId, firstId);
    assert.equal(state.streamGaps[0].endMs, 1500);
    assert.equal(state.droppedFrameCount, 20);
    assert.deepEqual(
      first.segments.map((item) => item.expertValid),
      [true, false],
    );
    state = command(state, { type: 'tick', now: now + 10000 });
    assert.equal(state.droppedFrameCount, 20);
    assert.equal(workflowLatest(state), first);
    state = command(state, { type: 'runtime', action: 'stream_reconnect', now: now + 11000 });
    state = command(state, { type: 'runtime', action: 'stream_disconnect', now: now + 12000 });
    assert.equal(workflowLatest(state), first);
    assert.equal(state.streamGaps.length, 1, 'scene reset is not a recording gap');
    state = command(state, {
      type: 'next-episode',
      sessionId,
      afterEpisodeId: firstId,
      now: now + 13000,
    });
    const secondId = workflowLatest(state).id;
    assert.equal(workflowLatest(state).segments[0].expertValid, false);
    assert.equal(state.streamGaps.length, 2);
    assert.equal(state.streamGaps[1].episodeId, secondId);
    assert.equal(state.streamGaps[1].startMs, 0);
    state = command(state, { type: 'runtime', action: 'stream_reconnect', now: now + 14000 });
    assert.equal(state.droppedFrameCount, 40);
    assert.equal(state.streamGaps[1].endMs, 1000);
    assert.deepEqual(
      workflowLatest(state).segments.map((item) => item.expertValid),
      [false, true],
    );
    state = command(state, {
      type: 'finish-episode',
      sessionId,
      episodeId: secondId,
      outcome: 'success',
      now: now + 15000,
    });
    assert.equal(workflowSession(state).episodes[0], first);
    assert.equal(workflowSession(state).durationMs, 3500);
    assert.equal(workflowLatest(state).outcome, 'success');
  });

  await test('runtime reset ends a between-episode session without losing sealed outcomes and annotations or accepting stale commands', () => {
    let state = prepare(createRobotDataWorkflow(context, now));
    const sessionId = state.data.activeSessionId;
    state = command(state, { type: 'start', sessionId, now: now + 3000 });
    state = command(state, { type: 'takeover', sessionId, now: now + 3500 });
    const episodeId = workflowLatest(state).id;
    state = command(state, {
      type: 'finish-episode',
      sessionId,
      episodeId,
      outcome: 'success',
      now: now + 4000,
    });
    const human = workflowLatest(state).segments.at(-1);
    state = command(state, {
      type: 'annotate',
      sessionId,
      episodeId,
      segmentId: human.id,
      annotation: {
        startFrame: human.startFrame,
        endFrame: human.endFrame,
        outcome: 'success',
        issues: [],
        intent: 'complete_task',
        note: '已完成。',
      },
    });
    const first = workflowLatest(state);
    const generation = state.runtimeGeneration;
    state = command(state, { type: 'reset-runtime', now: now + 10000 });
    assert.equal(state.runtimeSessionId, null);
    assert.equal(state.robot.phase, 'idle');
    assert.equal(workflowSession(state, sessionId).status, 'completed');
    assert.equal(workflowLatest(state, sessionId), first);
    assert.equal(first.outcome, 'success');
    assert.equal(first.segments.at(-1).annotations[0].note, '已完成。');
    assert.equal(
      command(state, {
        type: 'next-episode',
        sessionId,
        afterEpisodeId: episodeId,
        now: now + 11000,
      }),
      state,
    );
    assert.equal(
      command(state, {
        type: 'finish-episode',
        sessionId,
        episodeId,
        outcome: 'failure',
        now: now + 11000,
      }),
      state,
    );
    assert.equal(
      command(state, { type: 'runtime', action: 'eval_ready', generation, now: now + 11000 }),
      state,
    );
    let active = prepare(createRobotDataWorkflow(context, now));
    const activeId = active.data.activeSessionId;
    active = command(active, { type: 'start', sessionId: activeId, now: now + 3000 });
    active = command(active, { type: 'reset-runtime', now: now + 4000 });
    assert.equal(workflowLatest(active, activeId).status, 'sealed');
    assert.equal(workflowLatest(active, activeId).outcome, 'unknown');
  });
} finally {
  await server.close();
}
