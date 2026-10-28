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
    annotationValidationError,
  } = await server.ssrLoadModule('/src/dataReturnSimulation.ts');
  const { createRobotDataWorkflow, robotDataWorkflowReducer: workflowReduce } =
    await server.ssrLoadModule('/src/robotDataWorkflowSimulation.ts');
  const now = 2_000_000;
  const context = {
    deploymentRecordId: 'record-annotation',
    name: '标注示例',
    machine: '示例机器',
  };
  const initial = () => create(context, now);
  const locate = (state, segmentId = 'mock-segment-2') => {
    for (const session of state.sessions)
      for (const episode of session.episodes) {
        const segment = episode.segments.find((item) => item.id === segmentId);
        if (segment) return { session, episode, segment };
      }
    throw new Error(`Missing fixture ${segmentId}`);
  };
  const target = (state, segmentId) => {
    const { session, episode, segment } = locate(state, segmentId);
    return { sessionId: session.id, episodeId: episode.id, segmentId: segment.id };
  };
  const input = (overrides = {}) => ({
    startFrame: 250,
    endFrame: 300,
    outcome: 'failure',
    issues: ['grasp_failure'],
    intent: 'regrasp',
    note: '夹取偏移，重新夹取。',
    ...overrides,
  });
  const save = (state, annotation, annotationId, segmentId) =>
    reduce(state, { type: 'annotate', ...target(state, segmentId), annotation, annotationId });

  await test('seed annotations are meaningful and included in the original immutable snapshot', () => {
    const state = initial();
    const { segment } = locate(state);
    assert.equal(segment.annotations.length, 1);
    assert.equal(annotationValidationError(segment, segment.annotations[0]), undefined);
    assert.deepEqual(state.versions[0].segments[0].annotations, segment.annotations);
    assert.notEqual(state.versions[0].segments[0].annotations, segment.annotations);
    for (const session of state.sessions)
      for (const episode of session.episodes)
        for (const item of episode.segments) assert.ok(Array.isArray(item.annotations));
  });

  await test('annotations save, edit and remove with stable ids, copied input and no supervision changes', () => {
    let state = initial();
    const before = locate(state).segment;
    const value = input({ issues: ['pose_error', 'grasp_failure'], note: '  重新夹取  ' });
    state = save(state, value);
    let segment = locate(state).segment;
    const added = segment.annotations.at(-1);
    assert.match(added.id, /^mock-annotation-/);
    assert.equal(added.note, '重新夹取');
    value.issues.push('collision_risk');
    assert.deepEqual(added.issues, ['grasp_failure', 'pose_error']);
    assert.deepEqual(
      [segment.source, segment.expertValid, segment.review],
      [before.source, before.expertValid, before.review],
    );
    const sequence = state.nextAnnotationSequence;
    state = save(state, input({ outcome: 'success' }), added.id);
    segment = locate(state).segment;
    assert.equal(segment.annotations.at(-1).id, added.id);
    assert.equal(segment.annotations.at(-1).outcome, 'success');
    assert.equal(segment.annotations.length, 2);
    assert.equal(state.nextAnnotationSequence, sequence);
    assert.equal(save(state, input({ outcome: 'success' }), added.id), state);
    assert.equal(save(state, input(), 'stale-id'), state);
    state = reduce(state, { type: 'remove-annotation', ...target(state), annotationId: added.id });
    assert.equal(locate(state).segment.annotations.length, 1);
    assert.equal(state.nextAnnotationSequence, sequence);
    assert.equal(
      reduce(state, { type: 'remove-annotation', ...target(state), annotationId: added.id }),
      state,
    );
    state = save(state, input());
    assert.notEqual(locate(state).segment.annotations.at(-1).id, added.id);
    assert.equal(before.annotations.length, 1, 'the old state remains unchanged');
  });

  await test('invalid values, empty labels and invalid frame windows are rejected without state changes', () => {
    const state = initial();
    const { segment } = locate(state);
    const before = JSON.stringify(state);
    const invalid = [
      null,
      'text',
      {},
      input({ startFrame: 239 }),
      input({ endFrame: 481 }),
      input({ endFrame: 250 }),
      input({ startFrame: 250.5 }),
      input({ startFrame: NaN }),
      input({ endFrame: Infinity }),
      input({ outcome: 'maybe' }),
      input({ outcome: null }),
      input({ issues: 'pose_error' }),
      input({ issues: ['fabricated'] }),
      input({ issues: ['pose_error', 'pose_error'] }),
      input({ issues: [null] }),
      input({ issues: Array(1) }),
      input({ intent: 'operate_robot' }),
      input({ note: null }),
      input({ note: 'x'.repeat(241) }),
      input({ outcome: 'unknown', issues: [], intent: 'none', note: '   ' }),
    ];
    for (const value of invalid) {
      assert.equal(typeof annotationValidationError(segment, value), 'string');
      assert.equal(save(state, value), state);
    }
    for (const value of [
      input({ outcome: 'success', issues: [], intent: 'none', note: '' }),
      input({ outcome: 'unknown', issues: ['occlusion'], intent: 'none', note: '' }),
      input({ outcome: 'unknown', issues: [], intent: 'align_pose', note: '' }),
      input({ outcome: 'unknown', issues: [], intent: 'none', note: '仅有文本也有意义' }),
      input({ startFrame: segment.startFrame, endFrame: segment.endFrame, note: 'x'.repeat(240) }),
    ])
      assert.equal(annotationValidationError(segment, value), undefined);
    assert.equal(JSON.stringify(state), before);
  });

  await test('only sealed nonempty targets permit annotation, with missing targets and stale deletion ignored', () => {
    let state = initial();
    for (const override of [
      { sessionId: 'missing' },
      { episodeId: 'missing' },
      { segmentId: 'missing' },
    ]) {
      assert.equal(
        reduce(state, { type: 'annotate', ...target(state), ...override, annotation: input() }),
        state,
      );
    }
    const id = state.activeSessionId;
    state = reduce(state, { type: 'start', sessionId: id, now });
    const emptySegmentId = state.sessions.at(-1).episodes[0].segments[0].id;
    assert.equal(
      save(state, input({ startFrame: 0, endFrame: 1 }), undefined, emptySegmentId),
      state,
    );
    state = reduce(state, { type: 'tick', now: now + 1000 });
    assert.equal(
      save(state, input({ startFrame: 0, endFrame: 10 }), undefined, emptySegmentId),
      state,
    );
    state = reduce(state, { type: 'pause', sessionId: id, now: now + 1100 });
    assert.equal(
      save(state, input({ startFrame: 0, endFrame: 10 }), undefined, emptySegmentId),
      state,
    );
    state = reduce(state, { type: 'finish', sessionId: id, now: now + 1200 });
    state = save(state, input({ startFrame: 0, endFrame: 10 }), undefined, emptySegmentId);
    assert.equal(locate(state, emptySegmentId).segment.annotations.length, 1);
    let zero = initial();
    zero = reduce(zero, { type: 'start', sessionId: zero.activeSessionId, now });
    zero = reduce(zero, { type: 'finish', sessionId: zero.activeSessionId, now });
    const zeroId = zero.sessions.at(-1).episodes[0].segments[0].id;
    assert.equal(save(zero, input({ startFrame: 0, endFrame: 1 }), undefined, zeroId), zero);
  });

  await test('policy and invalid human annotations do not grant expert validity or publication eligibility', () => {
    let state = initial();
    const approved = dataReturnStats(state).approvedFrameCount;
    for (const id of ['mock-segment-1', 'mock-segment-6']) {
      const source = locate(state, id).segment;
      state = save(
        state,
        input({ startFrame: source.startFrame, endFrame: source.endFrame, outcome: 'success' }),
        undefined,
        id,
      );
      const labeled = locate(state, id).segment;
      assert.equal(labeled.annotations.length, 1);
      assert.equal(labeled.source, source.source);
      assert.equal(labeled.expertValid, false);
      assert.equal(labeled.review, source.review);
      state = reduce(state, { type: 'review', ...target(state, id), review: 'retained' });
    }
    assert.equal(dataReturnStats(state).approvedFrameCount, approved);
    assert.equal(
      reduce(state, { type: 'publish', segmentIds: ['mock-segment-1', 'mock-segment-6'], now }),
      state,
    );
  });

  await test('version identity includes deterministic structured labels and all nested snapshot data stays immutable', () => {
    let state = initial();
    const v1 = state.versions[0];
    const before = JSON.stringify(v1);
    const annotation = locate(state).segment.annotations[0];
    const { id, ...seedInput } = annotation;
    assert.equal(reduce(state, { type: 'publish', now }), state);
    state = save(state, { ...seedInput, note: '标注修订：对齐姿态后成功。' }, id);
    state = reduce(state, { type: 'publish', now: now + 100 });
    const v2 = state.versions[1];
    assert.notEqual(v1.contentKey, v2.contentKey);
    assert.equal(v2.segments[0].annotations[0].note, '标注修订：对齐姿态后成功。');
    assert.equal(reduce(state, { type: 'publish', name: '重复发布', now: now + 200 }), state);
    assert.equal(JSON.stringify(v1), before);
    for (const value of [
      v1.segments[0].annotations,
      v1.segments[0].annotations[0],
      v1.segments[0].annotations[0].issues,
    ])
      assert.ok(Object.isFrozen(value));
    assert.throws(() => {
      v1.segments[0].annotations[0].issues.push('other');
    }, TypeError);
    assert.throws(() => {
      v1.segments[0].annotations[0].note = '改写';
    }, TypeError);
    assert.throws(() => {
      v1.segments[0].annotations.push(annotation);
    }, TypeError);
    const extra = input({ issues: ['occlusion', 'grasp_failure'] });
    state = save(state, extra);
    const extraId = locate(state).segment.annotations.at(-1).id;
    state = reduce(state, { type: 'publish', now: now + 300 });
    const count = state.versions.length;
    state = save(state, { ...extra, issues: ['grasp_failure', 'occlusion'] }, extraId);
    state = reduce(state, { type: 'publish', now: now + 400 });
    assert.equal(state.versions.length, count, 'issue selection order is not a label revision');
    state = reduce(state, { type: 'remove-annotation', ...target(state), annotationId: extraId });
    state = save(state, seedInput, id);
    state = reduce(state, { type: 'publish', now: now + 500 });
    assert.equal(state.versions.length, count);
    assert.equal(
      state.lastPublishedVersionId,
      v1.id,
      'returning to identical labels reuses the original version',
    );
    state = reduce(state, { type: 'remove-annotation', ...target(state), annotationId: id });
    state = save(state, seedInput);
    assert.notEqual(locate(state).segment.annotations[0].id, id);
    state = reduce(state, { type: 'publish', now: now + 600 });
    assert.equal(
      state.lastPublishedVersionId,
      v1.id,
      'generated ids do not change semantic annotation content',
    );
    assert.equal(JSON.stringify(v1), before);
  });

  await test('workflow delegates historical annotations without activating runtime, and sensor splits start unannotated', () => {
    let state = createRobotDataWorkflow(context, now);
    const robot = state.robot;
    state = workflowReduce(state, { type: 'annotate', ...target(state.data), annotation: input() });
    const annotationId = locate(state.data).segment.annotations.at(-1).id;
    assert.equal(locate(state.data).segment.annotations.length, 2);
    assert.equal(state.robot, robot);
    assert.equal(state.runtimeSessionId, null);
    state = workflowReduce(state, {
      type: 'remove-annotation',
      ...target(state.data),
      annotationId,
    });
    assert.equal(locate(state.data).segment.annotations.length, 1);
    state = workflowReduce(state, { type: 'runtime', action: 'start_sdk', now: now + 10 });
    const generation = state.runtimeGeneration;
    state = workflowReduce(state, {
      type: 'runtime',
      action: 'sdk_ready',
      generation,
      now: now + 1110,
    });
    state = workflowReduce(state, { type: 'runtime', action: 'start_eval', now: now + 1120 });
    state = workflowReduce(state, {
      type: 'runtime',
      action: 'eval_ready',
      generation,
      now: now + 2820,
    });
    const sessionId = state.data.activeSessionId;
    state = workflowReduce(state, { type: 'start', sessionId, now: now + 2830 });
    state = workflowReduce(state, { type: 'takeover', sessionId, now: now + 3000 });
    state = workflowReduce(state, {
      type: 'runtime',
      action: 'stream_disconnect',
      now: now + 3500,
    });
    state = workflowReduce(state, { type: 'runtime', action: 'stream_reconnect', now: now + 4000 });
    const segments = state.data.sessions.find((session) => session.id === sessionId).episodes[0]
      .segments;
    assert.ok(segments.length >= 4);
    assert.ok(segments.every((segment) => segment.annotations.length === 0));
    const liveTarget = target(state.data, segments.at(-1).id);
    const before = state;
    state = workflowReduce(state, {
      type: 'annotate',
      ...liveTarget,
      annotation: input({
        startFrame: segments.at(-1).startFrame,
        endFrame: segments.at(-1).startFrame + 1,
      }),
    });
    assert.equal(state, before);
  });
} finally {
  await server.close();
}
