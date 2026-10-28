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
    createRobotDataWorkflow: create,
    robotDataWorkflowReducer: command,
    workflowStartReason,
    workflowCaptureEntry,
    workflowResumeReason,
    workflowStatus,
    activeWorkflowSession,
  } = await server.ssrLoadModule('/src/robotDataWorkflowSimulation.ts');
  const context = { deploymentRecordId: 'parent-a', name: '示例部署', machine: '示例机器' };
  const initial = () => create(context, 1_000_000);
  const draft = (state) => state.data.sessions.find((session) => session.status === 'draft');
  const session = (state, id) => state.data.sessions.find((value) => value.id === id);
  const prepare = (state) => {
    let now = state.robot.lastTickAt;
    state = command(state, { type: 'runtime', action: 'start_sdk', now: now + 10 });
    const generation = state.runtimeGeneration;
    state = command(state, { type: 'runtime', action: 'sdk_ready', generation, now: now + 1010 });
    state = command(state, { type: 'runtime', action: 'start_eval', now: now + 1020 });
    return command(state, { type: 'runtime', action: 'eval_ready', generation, now: now + 3020 });
  };
  const start = (state, id = draft(state).id) =>
    command(state, { type: 'start', sessionId: id, now: state.robot.lastTickAt + 10 });

  await test('capture entry exposes each preparation action locally without starting collection automatically', () => {
    let state = initial();
    const id = draft(state).id;
    let entry = workflowCaptureEntry(state, id);
    assert.equal(entry.action, 'start_sdk');
    assert.equal(entry.disabled, false);
    assert.match(entry.label, /启动 SDK/);
    assert.match(entry.hint, /同一位置/);
    state = command(state, { type: 'runtime', action: entry.action, now: 1_000_010 });
    entry = workflowCaptureEntry(state, id);
    assert.equal(entry.action, null);
    assert.equal(entry.disabled, true);
    assert.equal(entry.busy, true);
    assert.equal(session(state, id).status, 'draft');
    const generation = state.runtimeGeneration;
    state = command(state, {
      type: 'runtime',
      action: 'sdk_ready',
      generation,
      now: 1_001_010,
    });
    entry = workflowCaptureEntry(state, id);
    assert.equal(entry.action, 'start_eval');
    assert.equal(entry.disabled, false);
    assert.match(entry.label, /初始化 Eval/);
    assert.equal(command(state, { type: 'start', sessionId: id, now: 1_001_015 }), state);
    state = command(state, { type: 'runtime', action: entry.action, now: 1_001_020 });
    entry = workflowCaptureEntry(state, id);
    assert.equal(entry.action, null);
    assert.equal(entry.disabled, true);
    assert.equal(entry.busy, true);
    state = command(state, {
      type: 'runtime',
      action: 'eval_ready',
      generation,
      now: 1_003_020,
    });
    entry = workflowCaptureEntry(state, id);
    assert.equal(entry.action, 'start');
    assert.equal(entry.disabled, false);
    assert.equal(entry.busy, false);
    assert.equal(session(state, id).status, 'draft');
    assert.equal(state.robot.motionMs, 0);
    state = command(state, { type: entry.action, sessionId: id, now: 1_003_030 });
    assert.equal(session(state, id).status, 'running');
    assert.equal(state.runtimeSessionId, id);
  });

  await test('capture entry cannot prepare invalid, completed or competing sessions, including while paused', () => {
    let state = initial();
    for (const id of [undefined, 'missing']) {
      const entry = workflowCaptureEntry(state, id);
      assert.equal(entry.action, null);
      assert.equal(entry.disabled, true);
      assert.match(entry.hint, /请选择/);
    }
    const completed = state.data.sessions.find((item) => item.status === 'completed');
    assert.match(workflowCaptureEntry(state, completed.id).hint, /已开始/);
    assert.equal(workflowCaptureEntry(state, completed.id).action, null);
    state = start(prepare(state));
    const activeId = state.runtimeSessionId;
    state = command(state, {
      type: 'create-session',
      mode: 'dagger',
      now: state.robot.lastTickAt + 10,
    });
    const otherId = state.data.activeSessionId;
    for (const paused of [false, true]) {
      if (paused)
        state = command(state, {
          type: 'pause',
          sessionId: activeId,
          now: state.robot.lastTickAt + 10,
        });
      const entry = workflowCaptureEntry(state, otherId);
      assert.equal(entry.disabled, true);
      assert.equal(entry.busy, false);
      assert.equal(entry.action, null);
      assert.match(entry.hint, /结束当前会话/);
      assert.equal(
        command(state, { type: 'start', sessionId: otherId, now: state.robot.lastTickAt + 10 }),
        state,
      );
    }
  });

  await test('a stopped run offers SDK preparation for a new draft and reset invalidates the old completion', () => {
    let state = start(prepare(initial()));
    const id = state.runtimeSessionId;
    state = command(state, { type: 'finish', sessionId: id, now: state.robot.lastTickAt + 500 });
    state = command(state, {
      type: 'create-session',
      mode: 'dagger',
      now: state.robot.lastTickAt + 10,
    });
    const newId = state.data.activeSessionId;
    const entry = workflowCaptureEntry(state, newId);
    assert.equal(entry.action, 'start_sdk');
    assert.equal(entry.disabled, false);
    state = command(state, {
      type: 'runtime',
      action: entry.action,
      now: state.robot.lastTickAt + 10,
    });
    const generation = state.runtimeGeneration;
    state = command(state, { type: 'reset-runtime', now: state.robot.lastTickAt + 10 });
    assert.equal(
      command(state, {
        type: 'runtime',
        action: 'sdk_ready',
        generation,
        now: state.robot.lastTickAt + 1100,
      }),
      state,
    );
    assert.equal(session(state, newId).status, 'draft');
    assert.equal(workflowCaptureEntry(state, newId).action, 'start_sdk');
    assert.equal(session(state, id).status, 'completed');
  });

  await test('SDK and Eval preparation gate new capture while existing example data remains reviewable', () => {
    let state = initial();
    const id = draft(state).id;
    const before = JSON.stringify(state);
    assert.match(workflowStartReason(state, id), /SDK/);
    assert.equal(command(state, { type: 'start', sessionId: id, now: 1_000_010 }), state);
    assert.equal(command(state, { type: 'runtime', action: 'start_eval', now: 1_000_010 }), state);
    assert.equal(state.data.versions.length, 1);
    assert.ok(
      state.data.sessions.some((item) =>
        item.episodes.some((episode) => episode.status === 'sealed'),
      ),
    );
    assert.equal(JSON.stringify(state), before);
    state = prepare(state);
    assert.equal(state.robot.phase, 'ready');
    assert.equal(workflowStatus(state).source, 'none');
    assert.equal(workflowStartReason(state, id), undefined);
    assert.equal(state.robot.motionMs, 0);
    state = start(state, id);
    assert.equal(state.robot.phase, 'running');
    assert.equal(activeWorkflowSession(state).id, id);
    assert.equal(state.runtimeSessionId, id);
    assert.ok(state.robot.logs.at(-1).message.includes(session(state, id).name));
  });

  await test('the same run produces capture, human correction, review and an immutable version', () => {
    let state = prepare(initial());
    const id = draft(state).id;
    state = start(state, id);
    const now = state.robot.lastTickAt;
    state = command(state, { type: 'takeover', sessionId: id, now: now + 1000 });
    assert.equal(workflowStatus(state).source, 'human');
    assert.equal(
      state.robot.phase,
      'running',
      'human motion continues; policy is not the control owner',
    );
    state = command(state, { type: 'tick', now: now + 2400 });
    state = command(state, { type: 'resume', sessionId: id, now: now + 2500 });
    assert.equal(workflowStatus(state).source, 'policy');
    state = command(state, { type: 'finish', sessionId: id, now: now + 3000 });
    assert.equal(session(state, id).status, 'completed');
    assert.equal(state.robot.phase, 'stopped');
    assert.equal(activeWorkflowSession(state), undefined);
    const episode = session(state, id).episodes[0];
    assert.equal(episode.status, 'sealed');
    assert.deepEqual(
      episode.segments.map((part) => part.source),
      ['policy', 'human', 'policy'],
    );
    const human = episode.segments[1];
    assert.ok(human.endFrame > human.startFrame);
    state = command(state, {
      type: 'review',
      sessionId: id,
      episodeId: episode.id,
      segmentId: human.id,
      review: 'retained',
    });
    state = command(state, { type: 'publish', segmentIds: [human.id], now: now + 3100 });
    const version = state.data.versions.at(-1);
    assert.equal(version.segments.length, 1);
    assert.equal(version.segments[0].sessionId, id);
    assert.equal(version.context.deploymentRecordId, context.deploymentRecordId);
    assert.equal(version.simulationOnly, true);
    assert.ok(Object.isFrozen(version));
    const versionBefore = JSON.stringify(version);
    state = command(state, {
      type: 'review',
      sessionId: id,
      episodeId: episode.id,
      segmentId: human.id,
      review: 'excluded',
    });
    assert.equal(JSON.stringify(version), versionBefore);
    assert.ok(state.robot.logs.at(-1).message.includes(session(state, id).name));
    assert.match(state.robot.logs.at(-1).message, /封存/);
  });

  await test('pause freezes capture and motion, while another session cannot borrow or resume the runtime', () => {
    let state = start(prepare(initial()));
    const id = state.runtimeSessionId;
    const now = state.robot.lastTickAt;
    state = command(state, { type: 'pause', sessionId: id, now: now + 700 });
    const frames = session(state, id).frames;
    const motion = state.robot.motionMs;
    state = command(state, { type: 'tick', now: now + 4700 });
    assert.equal(session(state, id).frames, frames);
    assert.equal(state.robot.motionMs, motion);
    assert.equal(workflowStatus(state).source, 'hold');
    assert.equal(workflowResumeReason(state, id), undefined);
    state = command(state, { type: 'create-session', mode: 'dagger', now: now + 4700 });
    const other = state.data.activeSessionId;
    assert.notEqual(other, id);
    assert.match(workflowStartReason(state, other), /结束当前/);
    assert.equal(command(state, { type: 'start', sessionId: other, now: now + 4800 }), state);
    assert.equal(command(state, { type: 'finish', sessionId: other, now: now + 4800 }), state);
    state = command(state, { type: 'resume', sessionId: id, now: now + 4800 });
    state = command(state, { type: 'tick', now: now + 5100 });
    assert.equal(state.robot.motionMs, motion + 300);
    assert.equal(session(state, id).frames, frames + 6);
  });

  await test('runtime reset seals capture without erasing history and rejects stale initialization callbacks', () => {
    let state = initial();
    state = command(state, {
      type: 'runtime',
      action: 'start_sdk',
      now: state.robot.lastTickAt + 10,
    });
    const oldGeneration = state.runtimeGeneration;
    state = command(state, { type: 'reset-runtime', now: state.robot.lastTickAt + 10 });
    state = command(state, {
      type: 'runtime',
      action: 'start_sdk',
      now: state.robot.lastTickAt + 10,
    });
    const now = state.robot.lastTickAt;
    assert.equal(
      command(state, {
        type: 'runtime',
        action: 'sdk_ready',
        generation: oldGeneration,
        now: now + 100,
      }),
      state,
    );
    state = command(state, {
      type: 'runtime',
      action: 'sdk_ready',
      generation: state.runtimeGeneration,
      now: now + 100,
    });
    state = command(state, { type: 'runtime', action: 'start_eval', now: now + 110 });
    const evalGeneration = state.runtimeGeneration;
    state = command(state, { type: 'reset-runtime', now: now + 120 });
    assert.equal(
      command(state, {
        type: 'runtime',
        action: 'eval_ready',
        generation: evalGeneration,
        now: now + 200,
      }),
      state,
    );
    state = start(prepare(state));
    const id = state.runtimeSessionId;
    const seedVersion = state.data.versions[0];
    state = command(state, { type: 'reset-runtime', now: state.robot.lastTickAt + 1500 });
    assert.equal(state.robot.phase, 'idle');
    assert.equal(state.runtimeSessionId, null);
    assert.equal(session(state, id).status, 'completed');
    assert.equal(session(state, id).episodes[0].status, 'sealed');
    assert.equal(state.data.versions[0], seedVersion);
    assert.equal(state.data.sessions.length, 3);
    const oldCount = state.data.sessions.length;
    state = command(state, {
      type: 'create-session',
      mode: 'dagger',
      now: state.robot.lastTickAt + 10,
    });
    assert.equal(state.data.sessions.length, oldCount + 1);
    state = command(state, { type: 'reset', now: state.robot.lastTickAt + 20 });
    assert.equal(state.data.sessions.length, 3);
    assert.equal(state.robot.phase, 'idle');
    assert.equal(state.data.context.deploymentRecordId, context.deploymentRecordId);
  });

  await test('stream outages freeze only the picture, count missing frames and invalidate affected expert windows', () => {
    let state = start(prepare(initial()));
    const id = state.runtimeSessionId;
    const now = state.robot.lastTickAt;
    state = command(state, { type: 'takeover', sessionId: id, now: now + 500 });
    state = command(state, { type: 'runtime', action: 'stream_disconnect', now: now + 1000 });
    const picture = state.robot.frameMotionMs;
    const motion = state.robot.motionMs;
    const frames = session(state, id).frames;
    state = command(state, { type: 'tick', now: now + 2000 });
    assert.equal(state.robot.frameMotionMs, picture);
    assert.equal(state.robot.motionMs, motion + 1000);
    assert.equal(session(state, id).frames, frames + 20);
    assert.equal(state.droppedFrameCount, 20);
    assert.equal(workflowStatus(state).source, 'human');
    state = command(state, { type: 'runtime', action: 'stream_reconnect', now: now + 2500 });
    assert.ok(state.robot.frameMotionMs > picture);
    assert.equal(state.droppedFrameCount, 30);
    assert.equal(state.streamGaps[0].endMs - state.streamGaps[0].startMs, 1500);
    state = command(state, { type: 'finish', sessionId: id, now: now + 3000 });
    const episode = session(state, id).episodes[0];
    const human = episode.segments.filter((part) => part.source === 'human');
    assert.deepEqual(
      human.map((part) => part.expertValid),
      [true, false, true],
    );
    for (const part of episode.segments)
      state = command(state, {
        type: 'review',
        sessionId: id,
        episodeId: episode.id,
        segmentId: part.id,
        review: 'retained',
      });
    state = command(state, {
      type: 'publish',
      segmentIds: episode.segments.map((part) => part.id),
      now: now + 3100,
    });
    assert.deepEqual(
      state.data.versions.at(-1).sourceSegmentIds,
      human.filter((part) => part.expertValid).map((part) => part.id),
    );
  });

  await test('shadow collection stays human-owned with zero takeovers and records unavailable initial camera frames', () => {
    let state = prepare(initial());
    state = command(state, {
      type: 'create-session',
      mode: 'shadow',
      now: state.robot.lastTickAt + 10,
    });
    const id = state.data.activeSessionId;
    state = command(state, {
      type: 'runtime',
      action: 'stream_disconnect',
      now: state.robot.lastTickAt + 10,
    });
    state = start(state, id);
    const now = state.robot.lastTickAt;
    assert.equal(workflowStatus(state).source, 'human');
    assert.equal(session(state, id).takeovers, 0);
    assert.match(state.robot.logs.at(-1).message, /影子建议/);
    assert.equal(session(state, id).episodes[0].segments[0].expertValid, false);
    assert.equal(command(state, { type: 'takeover', sessionId: id, now: now + 10 }), state);
    state = command(state, { type: 'tick', now: now + 1000 });
    assert.equal(state.droppedFrameCount, 20);
    state = command(state, { type: 'finish', sessionId: id, now: now + 1500 });
    assert.equal(state.streamGaps[0].endMs, 1500);
    assert.equal(session(state, id).takeovers, 0);
  });
} finally {
  await server.close();
}
