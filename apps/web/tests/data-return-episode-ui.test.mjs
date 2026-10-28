import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});
const freeze = (value) => {
  if (value && typeof value === 'object') {
    for (const item of Object.values(value)) freeze(item);
    Object.freeze(value);
  }
  return value;
};
const buttons = (html) =>
  [...html.matchAll(/<button\b[^>]*>[\s\S]*?<\/button>/g)].map((match) => ({
    html: match[0],
    text: /aria-label="([^"]+)"/.exec(match[0])?.[1] || match[0].replace(/<[^>]*>/g, '').trim(),
  }));
const noExternalOperations = (html) => {
  assert.doesNotMatch(html, /<(?:iframe|video|img)\b[^>]*\bsrc=/);
  for (const button of buttons(html))
    assert.doesNotMatch(button.text, /启动 SDK|初始化 Eval|真实复位|启动服务|停止服务/);
};

try {
  const { default: EpisodeControl } = await server.ssrLoadModule(
    '/src/DataReturnEpisodeControl.tsx',
  );
  const { createDataReturnSimulation } = await server.ssrLoadModule('/src/dataReturnSimulation.ts');
  const state = createDataReturnSimulation(
    { deploymentRecordId: 'record-episode-preview', name: 'Episode 演示', machine: '模拟机器' },
    1_000_000,
  );
  const completed = state.sessions[0];
  const baseEpisode = completed.episodes[0];
  const recording = {
    ...baseEpisode,
    id: 'episode-current',
    name: 'Episode 002',
    status: 'recording',
    outcome: 'unknown',
    startedAt: 990_000,
    finishedAt: undefined,
  };
  const callbacks = {
    onFinishEpisode() {
      assert.fail('SSR must not decide an outcome or finish an Episode');
    },
    onStartNext() {
      assert.fail('SSR must not start another Episode');
    },
    onSelectEpisode() {
      assert.fail('SSR must not change replay selection');
    },
  };
  const render = (session, props = {}) =>
    renderToStaticMarkup(React.createElement(EpisodeControl, { session, ...callbacks, ...props }));
  const activeSession = (status = 'running', episode = recording) =>
    freeze({
      ...completed,
      status,
      finishedAt: undefined,
      episodes: [baseEpisode, episode],
    });

  await test('running, takeover, and paused Episodes expose explicit operator success, failure, and abort decisions', () => {
    for (const status of ['running', 'takeover', 'paused']) {
      const session = activeSession(status);
      const before = JSON.stringify(session);
      const html = render(session);
      assert.match(html, /aria-label="Episode 采集控制"/);
      const controls = buttons(html);
      for (const outcome of [/成功/, /失败/, /中止|未判定/]) {
        const action = controls.find((button) => outcome.test(button.text));
        assert.ok(action, `operator outcome control ${outcome} for ${status}`);
        assert.doesNotMatch(action.html, /disabled/);
      }
      assert.ok(!controls.some((button) => /开始 Episode/.test(button.text)));
      assert.match(html, /人工|操作[者员]/);
      assert.match(html, /由操作员判断任务目标是否完成/);
      assert.equal(JSON.stringify(session), before);
      assert.equal(session.episodes.at(-1).outcome, 'unknown');
      noExternalOperations(html);
    }
  });

  await test('between Episodes waits for a manual next-Episode action and preserves sealed history', () => {
    const session = activeSession('between_episodes', {
      ...recording,
      status: 'sealed',
      outcome: 'failure',
      finishedAt: 1_000_000,
    });
    const before = JSON.stringify(session);
    const html = render(session, { selectedEpisodeId: baseEpisode.id });
    const controls = buttons(html);
    const next = controls.find((button) => /开始 Episode 003/.test(button.text));
    assert.ok(next, 'the next number belongs to this session');
    assert.doesNotMatch(next.html, /disabled/);
    assert.ok(
      !controls.some((button) => /结束.*Episode|成功结束|失败结束|中止当前/.test(button.text)),
    );
    assert.match(html, /aria-label="本会话 Episodes"/);
    assert.match(html, /Episode 001/);
    assert.match(html, /Episode 002/);
    assert.match(html, /失败/);
    assert.match(html, /等待|手动|下一/);
    assert.equal(JSON.stringify(session), before);
    noExternalOperations(html);
  });

  await test('replaying an older Episode keeps current collection explicit and provides a way back to it', () => {
    const session = activeSession('takeover');
    const html = render(session, { selectedEpisodeId: baseEpisode.id });
    assert.match(html, /正在回放 Episode 001/);
    assert.match(html, /最新 Episode|仍在采集/);
    const current = buttons(html).find((button) => /查看当前采集/.test(button.text));
    assert.ok(current);
    assert.doesNotMatch(current.html, /disabled/);
    const replay = buttons(html).find((button) => button.text === '回放 Episode 001');
    assert.ok(replay);
    assert.doesNotMatch(replay.html, /disabled/);
    assert.match(html, /Episode 002/);
    assert.equal(session.status, 'takeover');
    noExternalOperations(html);
  });

  await test('a next-Episode blocker is visible and disables the manual start action', () => {
    const session = activeSession('between_episodes', {
      ...recording,
      status: 'sealed',
      outcome: 'unknown',
    });
    const html = render(session, { nextBlockedReason: '示例视频尚未恢复，请先恢复视频。' });
    const next = buttons(html).find((button) => /开始 Episode 003/.test(button.text));
    assert.ok(next);
    assert.match(next.html, /disabled/);
    assert.match(html, /示例视频尚未恢复，请先恢复视频。/);
    noExternalOperations(html);
  });

  await test('completed sessions show history without reopening collection or offering outcome decisions', () => {
    const session = freeze({ ...completed });
    const html = render(session, { selectedEpisodeId: baseEpisode.id });
    assert.match(html, /aria-label="本会话 Episodes"/);
    assert.match(html, /Episode 001/);
    assert.match(html, /Episode 002/);
    assert.ok(
      !buttons(html).some((button) => /开始 Episode|结束.*Episode|中止当前/.test(button.text)),
    );
    noExternalOperations(html);
  });

  await test('draft and missing-Episode sessions do not invent a current Episode or a completion action', () => {
    for (const session of [
      freeze({ ...completed, status: 'draft', episodes: [] }),
      freeze({ ...completed, status: 'running', episodes: [] }),
    ])
      assert.equal(render(session), '');
  });

  await test('zero-frame current Episodes remain explicit local operator decisions without automatic success', () => {
    const session = activeSession('paused', {
      ...recording,
      durationMs: 0,
      frames: 0,
      segments: [],
    });
    const before = JSON.stringify(session);
    const html = render(session);
    assert.match(html, /Episode 002/);
    assert.match(html, /0/);
    assert.equal(JSON.stringify(session), before);
    assert.equal(session.episodes.at(-1).outcome, 'unknown');
    noExternalOperations(html);
  });

  await test('Episode history escapes untrusted display names and does not mutate outcomes', () => {
    const session = freeze({
      ...completed,
      episodes: [
        { ...baseEpisode, name: '<img src="external" onerror="example">', outcome: 'unknown' },
      ],
    });
    const before = JSON.stringify(session);
    const html = render(session, { selectedEpisodeId: baseEpisode.id });
    assert.match(html, /&lt;img/);
    assert.doesNotMatch(html, /<img\b/);
    assert.match(html, /中止|未判定/);
    assert.equal(JSON.stringify(session), before);
    noExternalOperations(html);
  });
} finally {
  await server.close();
}
