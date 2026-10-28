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
    createDataReturnSimulation,
    dataReturnReducer: reduce,
    dataReturnStats,
  } = await server.ssrLoadModule('/src/dataReturnSimulation.ts');
  const now = Date.parse('2026-10-27T12:00:00Z');
  const context = Object.freeze({
    deploymentRecordId: 'record-fixture',
    name: '折叠衣服部署',
    machine: '示例机器',
    runtimeId: 'runtime-fixture',
  });
  const initial = () => createDataReturnSimulation(context, now);
  const active = (state) => state.sessions.find((session) => session.id === state.activeSessionId);
  const control = (state, type, offset) =>
    reduce(state, { type, sessionId: state.activeSessionId, now: now + offset });
  const review = (state, segmentId, value, note) => {
    for (const session of state.sessions)
      for (const episode of session.episodes) {
        if (episode.segments.some((segment) => segment.id === segmentId))
          return reduce(state, {
            type: 'review',
            sessionId: session.id,
            episodeId: episode.id,
            segmentId,
            review: value,
            note,
          });
      }
    throw new Error('Missing fixture segment');
  };

  await test('seed durations, frame boundaries, provenance and v1 agree', () => {
    const state = initial();
    assert.equal(state.versions.length, 1);
    assert.equal(state.versions[0].simulationOnly, true);
    assert.equal(state.versions[0].context.deploymentRecordId, context.deploymentRecordId);
    assert.deepEqual(state.versions[0].sourceSegmentIds, ['mock-segment-2']);
    assert.equal(state.versions[0].frameCount, 240);
    for (const session of state.sessions) {
      assert.equal(
        session.durationMs,
        session.episodes.reduce((sum, episode) => sum + episode.durationMs, 0),
      );
      assert.equal(
        session.frames,
        session.episodes.reduce((sum, episode) => sum + episode.frames, 0),
      );
      for (const episode of session.episodes) {
        assert.equal(episode.frames, Math.floor((episode.durationMs * episode.fps) / 1000));
        let end = 0;
        for (const segment of episode.segments) {
          assert.equal(segment.startFrame, end);
          assert.ok(segment.endFrame >= segment.startFrame);
          if (segment.source === 'policy') assert.equal(segment.expertValid, false);
          end = segment.endFrame;
        }
        assert.equal(end, episode.frames);
      }
    }
    assert.equal(dataReturnStats(state).frameCount, 2300);
    assert.equal(dataReturnStats(state).pendingExpertReviewCount, 1);
    assert.equal(dataReturnStats(state).approvedFrameCount, 240);
  });

  await test('DAgger transitions record separate policy and human spans, excluding paused time', () => {
    let state = reduce(initial(), { type: 'create-session', mode: 'dagger', now });
    const draftId = state.activeSessionId;
    state = control(state, 'start', 0);
    state = reduce(state, { type: 'tick', now: now + 2000 });
    state = control(state, 'takeover', 3000);
    state = reduce(state, { type: 'tick', now: now + 5000 });
    const stillRecording = state;
    const humanId = active(state).episodes[0].segments[1].id;
    assert.strictEqual(
      review(state, humanId, 'retained'),
      stillRecording,
      'live segments are not reviewable',
    );
    state = control(state, 'pause', 6000);
    assert.equal(dataReturnStats(state).activeSessionCount, 1);
    assert.equal(dataReturnStats(state).recordingSessionCount, 0);
    assert.strictEqual(reduce(state, { type: 'tick', now: now + 16000 }), state);
    state = control(state, 'resume', 16000);
    state = reduce(state, { type: 'tick', now: now + 17000 });
    state = control(state, 'finish', 18000);
    const session = active(state);
    assert.equal(session.id, draftId);
    assert.equal(session.status, 'completed');
    assert.equal(session.durationMs, 8000);
    assert.equal(session.frames, 160);
    assert.equal(session.takeovers, 1);
    assert.equal(session.finishedAt - session.startedAt, 18000);
    assert.equal(session.episodes[0].status, 'sealed');
    assert.deepEqual(
      session.episodes[0].segments.map((segment) => [
        segment.source,
        segment.startFrame,
        segment.endFrame,
        segment.expertValid,
      ]),
      [
        ['policy', 0, 60, false],
        ['human', 60, 120, true],
        ['policy', 120, 160, false],
      ],
    );
    state = review(state, humanId, 'retained', '人工纠错示例');
    state = reduce(state, { type: 'publish', segmentIds: [humanId], now: now + 19000 });
    assert.equal(state.versions.at(-1).frameCount, 60);
    assert.deepEqual(state.versions.at(-1).sourceSegmentIds, [humanId]);
  });

  await test('shadow mode remains human-controlled with policy suggestions and no takeover count', () => {
    let state = reduce(initial(), {
      type: 'create-session',
      mode: 'shadow',
      name: '影子会话',
      now,
    });
    state = control(state, 'start', 0);
    state = reduce(state, { type: 'tick', now: now + 1000 });
    assert.strictEqual(control(state, 'takeover', 1000), state);
    state = control(state, 'pause', 2000);
    state = control(state, 'resume', 10000);
    state = control(state, 'finish', 12000);
    const session = active(state);
    assert.equal(session.durationMs, 4000);
    assert.equal(session.takeovers, 0);
    assert.ok(
      session.episodes[0].segments.every(
        (segment) => segment.source === 'human' && segment.expertValid,
      ),
    );
    assert.equal(state.sessions.find((item) => item.id === 'mock-session-2').takeovers, 0);
  });

  await test('paused sessions still occupy the single control session and cannot lead to dual resume', () => {
    let state = control(initial(), 'start', 0);
    const firstId = state.activeSessionId;
    state = control(state, 'pause', 1000);
    state = reduce(state, { type: 'create-session', mode: 'shadow', now: now + 2000 });
    const secondId = state.activeSessionId;
    assert.strictEqual(control(state, 'start', 3000), state);
    state = reduce(state, { type: 'finish', sessionId: firstId, now: now + 4000 });
    state = reduce(state, { type: 'start', sessionId: secondId, now: now + 5000 });
    assert.equal(state.sessions.find((item) => item.id === secondId).status, 'running');
    assert.equal(dataReturnStats(state).activeSessionCount, 1);
    // Even a stale external view containing a second paused row cannot resume it.
    const stale = {
      ...state,
      sessions: state.sessions.map((item) =>
        item.id === firstId ? { ...item, status: 'paused' } : item,
      ),
    };
    assert.strictEqual(
      reduce(stale, { type: 'resume', sessionId: firstId, now: now + 6000 }),
      stale,
    );
  });

  await test('publication requires approved, positive-length valid expert spans; policy approval is insufficient', () => {
    let state = initial();
    state = review(state, 'mock-segment-1', 'retained', '仅用于分析');
    state = review(state, 'mock-segment-6', 'retained', '审核不能修复缺失的专家信号');
    assert.strictEqual(
      reduce(state, {
        type: 'publish',
        segmentIds: ['mock-segment-1', 'mock-segment-5', 'mock-segment-6'],
        now,
      }),
      state,
    );
    state = review(state, 'mock-segment-5', 'retained', '有效纠错');
    const published = reduce(state, {
      type: 'publish',
      segmentIds: ['mock-segment-1', 'mock-segment-5', 'mock-segment-6'],
      now,
    });
    assert.equal(published.versions.length, 2);
    assert.deepEqual(published.versions[1].sourceSegmentIds, ['mock-segment-5']);
    assert.equal(published.versions[1].frameCount, 200);
    assert.ok(
      published.versions[1].segments.every(
        (segment) =>
          segment.expertValid && segment.source === 'human' && segment.review === 'retained',
      ),
    );
    assert.strictEqual(reduce(published, { type: 'publish', segmentIds: [], now }), published);
    let empty = reduce(initial(), { type: 'create-session', mode: 'dagger', now });
    empty = control(empty, 'start', 0);
    empty = control(empty, 'takeover', 0);
    empty = control(empty, 'finish', 0);
    const zeroFrameHuman = active(empty).episodes[0].segments[1].id;
    empty = review(empty, zeroFrameHuman, 'retained');
    assert.strictEqual(
      reduce(empty, { type: 'publish', segmentIds: [zeroFrameHuman], now }),
      empty,
    );
  });

  await test('version snapshots never change after review edits, and reselecting the same content is idempotent', () => {
    let state = initial();
    const v1 = state.versions[0];
    const frozen = JSON.stringify(v1);
    state = review(state, 'mock-segment-5', 'retained', '第二段已审核');
    state = reduce(state, {
      type: 'publish',
      segmentIds: ['mock-segment-5', 'mock-segment-2'],
      name: '示例 v2',
      now,
    });
    const v2 = state.versions[1];
    assert.equal(v2.frameCount, 440);
    assert.strictEqual(
      reduce(state, {
        type: 'publish',
        segmentIds: ['mock-segment-2', 'mock-segment-5', 'mock-segment-2'],
        name: '重复点击',
        now: now + 1000,
      }),
      state,
    );
    state = review(state, 'mock-segment-2', 'excluded', '后续改变审核');
    state = review(state, 'mock-segment-5', 'excluded', '排除待修复');
    assert.strictEqual(state.versions[0], v1);
    assert.strictEqual(state.versions[1], v2);
    assert.equal(JSON.stringify(v1), frozen);
    assert.equal(v2.frameCount, 440);
    assert.equal(dataReturnStats(state).publishableSegmentCount, 0);
    assert.throws(() => {
      v1.segments[0].note = '不可改写';
    }, TypeError);
    assert.throws(() => {
      v2.sourceSegmentIds.push('fabricated');
    }, TypeError);
    assert.strictEqual(reduce(state, { type: 'publish', now: now + 2000 }), state);
  });

  await test('ticks are monotonic, completed sessions stay sealed, reset restores this context only', () => {
    let state = control(initial(), 'start', 0);
    state = reduce(state, { type: 'tick', now: now + 2000 });
    assert.strictEqual(reduce(state, { type: 'tick', now: now + 1000 }), state);
    assert.strictEqual(reduce(state, { type: 'tick', now: NaN }), state);
    state = control(state, 'finish', 3000);
    assert.strictEqual(control(state, 'resume', 4000), state);
    assert.strictEqual(reduce(state, { type: 'tick', now: now + 10000 }), state);
    const reset = reduce(state, { type: 'reset', now: now + 20000 });
    assert.deepEqual(reset.context, context);
    assert.equal(reset.sessions.length, 3);
    assert.equal(reset.versions.length, 1);
    assert.equal(reset.sessions.find((item) => item.id === 'mock-session-3').status, 'draft');
    assert.equal(dataReturnStats(reset).activeSessionCount, 0);
    assert.equal(
      reset.sessions.find((item) => item.id === 'mock-session-3').createdAt,
      now + 20000,
    );
  });
} finally {
  await server.close();
}
