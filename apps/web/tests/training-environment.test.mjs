import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});

try {
  const { trainingEnvironment, TrainingEnvironmentDetails } =
    await server.ssrLoadModule('/src/TrainingEnvironment.tsx');
  const source = (name, root) => ({
    code_dir: root,
    runtime_profile: {
      id: 'native', name, code_dir: root, python: `${root}/.venv/bin/python`,
      adapter: 'openpi-sft', execution_mode: 'native',
    },
  });
  const run = (snapshot) => ({
    id: 'training', name: 'Training', node_id: 'machine-a', profile_id: 'native',
    template_id: 'openpi-sft', state: 'QUEUED', created_at: '2026-11-03T00:00:00Z',
    code_expectation: snapshot,
  });

  await test('queued training shows the environment frozen during preflight', () => {
    const selected = run(source('User A OpenPI', '/projects/user-a'));
    assert.deepEqual(trainingEnvironment(selected), {
      id: 'native', name: 'User A OpenPI', codeDir: '/projects/user-a',
      python: '/projects/user-a/.venv/bin/python',
    });
    const html = renderToStaticMarkup(React.createElement('dl', {},
      React.createElement(TrainingEnvironmentDetails, { run: selected })));
    for (const value of ['运行环境', '环境标识', '代码目录', 'Python 环境', 'User A OpenPI',
      '/projects/user-a/.venv/bin/python']) assert.ok(html.includes(value), value);
  });

  await test('the latest execution snapshot takes priority over preflight and does not mix bindings', () => {
    const selected = run(source('Preflight environment', '/projects/old'));
    selected.attempts = [
      { id: 'first', code_snapshot: source('First attempt', '/projects/first') },
      { id: 'latest', code_snapshot: source('Actual environment', '/projects/actual') },
    ];
    assert.deepEqual(trainingEnvironment(selected), {
      id: 'native', name: 'Actual environment', codeDir: '/projects/actual',
      python: '/projects/actual/.venv/bin/python',
    });
    selected.attempts[1].code_snapshot = { code_dir: '/projects/drifted' };
    assert.deepEqual(trainingEnvironment(selected), {
      id: 'native', name: 'native', codeDir: '/projects/drifted', python: undefined,
    });
  });

  await test('same profile IDs on different machines retain their own frozen environments', () => {
    const first = run(source('Machine A environment', '/machine-a/code'));
    const second = { ...run(source('Machine B environment', '/machine-b/code')), node_id: 'machine-b' };
    assert.equal(trainingEnvironment(first).name, 'Machine A environment');
    assert.equal(trainingEnvironment(second).name, 'Machine B environment');
    assert.equal(trainingEnvironment(first).python, '/machine-a/code/.venv/bin/python');
    assert.equal(trainingEnvironment(second).python, '/machine-b/code/.venv/bin/python');
  });

  await test('legacy training retains its recorded directory and leaves unknown environment details explicit', () => {
    const selected = run({ code_dir: '/historical/code' });
    assert.deepEqual(trainingEnvironment(selected), {
      id: 'native', name: 'native', codeDir: '/historical/code', python: undefined,
    });
    const html = renderToStaticMarkup(React.createElement('dl', {},
      React.createElement(TrainingEnvironmentDetails, { run: selected })));
    assert.match(html, /Python 环境<\/dt><dd>未记录<\/dd>/);
    assert.ok(html.includes('/historical/code'));
    delete selected.code_expectation;
    assert.equal(trainingEnvironment(selected).codeDir, undefined);
  });
} finally {
  await server.close();
}
