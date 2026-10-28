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
    createRobotMock,
    transitionRobotMock: command,
    tickRobotMock: tick,
    ROBOT_MOCK_LOG_LIMIT,
  } = await server.ssrLoadModule('/src/robotDeploymentSimulation.ts');
  const ready = () => {
    let state = createRobotMock(1000);
    state = command(state, 'start_sdk', 1100);
    state = command(state, 'sdk_ready', 2000);
    state = command(state, 'start_eval', 2100);
    return command(state, 'eval_ready', 4000);
  };

  await test('the demo requires SDK then Eval, and initialization never starts policy execution', () => {
    let state = createRobotMock(1000);
    for (const action of ['start_eval', 'eval_ready', 'sdk_ready', 'start', 'pause']) {
      assert.equal(command(state, action, 1100), state, action);
    }
    state = command(state, 'start_sdk', 1100);
    assert.equal(command(state, 'start_sdk', 1200), state);
    assert.equal(command(state, 'start_eval', 1200), state);
    state = command(state, 'sdk_ready', 2000);
    assert.equal(state.telemetry.feedbackAt, 2000);
    assert.equal(state.frameAt, null);
    state = command(state, 'start_eval', 2100);
    const collecting = tick(state, 2600);
    assert.ok(collecting.telemetry.historyFrames > 0);
    assert.ok(collecting.telemetry.historyFrames < 241);
    state = command(collecting, 'eval_ready', 4000);
    assert.equal(state.phase, 'ready');
    assert.equal(state.telemetry.historyFrames, 241);
    assert.equal(state.motionMs, 0);
    assert.equal(tick(state, 12000).phase, 'ready');
    assert.equal(tick(state, 12000).motionMs, 0);
  });

  await test('simulated telemetry is repeatable and independent of browser tick frequency', () => {
    const initial = command(ready(), 'start', 4100);
    const before = JSON.stringify(initial);
    const oneTick = tick(initial, 6100);
    const manyTicks = [4400, 4700, 5200, 5800, 6100].reduce(tick, initial);
    assert.deepEqual(oneTick, manyTicks);
    assert.equal(JSON.stringify(initial), before, 'pure update must not mutate input');
    assert.notDeepEqual(oneTick.telemetry.joints, initial.telemetry.joints);
    assert.notDeepEqual(oneTick.telemetry.tcp, initial.telemetry.tcp);
    assert.equal(oneTick.telemetry.joints.length, 6);
    assert.equal(oneTick.telemetry.tcp.length, 6);
    assert.ok(oneTick.telemetry.gripperMm >= 0 && oneTick.telemetry.gripperMm <= 85);
    assert.ok(oneTick.telemetry.submittedSteps > 0);
  });

  await test('pause freezes motion; resume does not accumulate time spent paused', () => {
    const running = command(ready(), 'start', 4100);
    const paused = command(running, 'pause', 4600);
    const waiting = tick(paused, 14600);
    assert.equal(waiting.motionMs, 500);
    assert.deepEqual(waiting.telemetry.joints, paused.telemetry.joints);
    assert.deepEqual(waiting.telemetry.tcp, paused.telemetry.tcp);
    assert.equal(waiting.telemetry.gripperMm, paused.telemetry.gripperMm);
    assert.equal(waiting.frameMotionMs, paused.frameMotionMs);
    assert.equal(waiting.telemetry.fps, 12, 'camera remains available while execution pauses');
    assert.equal(waiting.telemetry.feedbackAt, 14600);
    const resumed = tick(command(waiting, 'start', 14600), 14850);
    assert.equal(resumed.motionMs, 750);
  });

  await test('video loss freezes only the displayed frame and does not claim execution stopped', () => {
    const running = command(ready(), 'start', 4100);
    const lost = command(running, 'stream_disconnect', 4600);
    const later = tick(lost, 6600);
    assert.equal(later.phase, 'running');
    assert.equal(later.streamConnected, false);
    assert.equal(later.telemetry.fps, 0);
    assert.equal(later.frameAt, lost.frameAt);
    assert.equal(later.frameMotionMs, lost.frameMotionMs);
    assert.ok(later.motionMs > lost.motionMs);
    assert.notDeepEqual(later.telemetry.tcp, lost.telemetry.tcp);
    assert.equal(later.telemetry.feedbackAt, 6600);
    assert.equal(command(later, 'stream_disconnect', 6700), later);
    const restored = command(later, 'stream_reconnect', 6800);
    assert.equal(restored.phase, 'running');
    assert.equal(restored.streamConnected, true);
    assert.equal(restored.frameAt, 6800);
    assert.equal(restored.frameMotionMs, restored.motionMs);
    assert.equal(restored.telemetry.fps, 12);
  });

  await test('ending a session keeps the final sample stale and rejects delayed startup events', () => {
    const stopped = command(command(ready(), 'start', 4100), 'stop', 5100);
    const later = tick(stopped, 20000);
    assert.equal(later.phase, 'stopped');
    assert.equal(later.telemetry.fps, 0);
    assert.equal(later.frameAt, stopped.frameAt);
    assert.equal(later.motionMs, stopped.motionMs);
    assert.deepEqual(later.telemetry, stopped.telemetry);
    for (const action of ['sdk_ready', 'eval_ready', 'start', 'start_sdk', 'start_eval', 'stop']) {
      assert.equal(command(later, action, 21000), later, action);
    }
    const starting = command(createRobotMock(0), 'start_sdk', 100);
    const cancelled = command(starting, 'stop', 200);
    assert.equal(command(cancelled, 'sdk_ready', 1000), cancelled);
  });

  await test('reset clears all simulated samples and old initialization callbacks cannot advance it', () => {
    const stopped = command(ready(), 'stop', 5000);
    const fresh = command(stopped, 'reset', 6000);
    assert.deepEqual(fresh, createRobotMock(6000));
    assert.equal(fresh.frameAt, null);
    assert.equal(fresh.telemetry.feedbackAt, null);
    assert.equal(fresh.logs.length, 0);
    assert.equal(command(fresh, 'eval_ready', 7000), fresh);
    assert.equal(command(fresh, 'sdk_ready', 7000), fresh);
  });

  await test('repeated monitoring events retain only bounded, uniquely ordered mock logs', () => {
    let state = ready();
    for (let i = 0; i < ROBOT_MOCK_LOG_LIMIT * 2; i++) {
      state = command(state, 'stream_disconnect', 5000 + i * 2);
      state = command(state, 'stream_reconnect', 5001 + i * 2);
    }
    assert.equal(state.logs.length, ROBOT_MOCK_LOG_LIMIT);
    assert.equal(new Set(state.logs.map((entry) => entry.id)).size, ROBOT_MOCK_LOG_LIMIT);
    assert.ok(state.logs.every((entry) => entry.message.startsWith('[MOCK]')));
    assert.ok(state.logs[0].id > 1);
  });

  await test('clock rollback and non-finite timestamps cannot reverse or corrupt telemetry', () => {
    const running = tick(command(ready(), 'start', 4100), 5100);
    for (const timestamp of [1000, NaN, Infinity, -Infinity]) {
      const state = tick(running, timestamp);
      assert.equal(state.motionMs, running.motionMs);
      assert.equal(state.lastTickAt, running.lastTickAt);
      assert.ok(state.telemetry.tcp.every(Number.isFinite));
      assert.ok(state.telemetry.joints.every(Number.isFinite));
    }
  });
} finally {
  await server.close();
}
