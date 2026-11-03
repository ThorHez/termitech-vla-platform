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
const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;
try {
  const { submitTrainingRun } = await server.ssrLoadModule('/src/runSubmission.ts');
  const transport = await server.ssrLoadModule('/src/api.ts');
  const options = (overrides = {}) => ({
    preflightId: 'preview-alice',
    confirmedConfigSha256: 'a'.repeat(64),
    idempotencyKey: 'run-submission-alice',
    finishDraft: async () => {},
    isActive: () => true,
    onSubmitted: (run) => transport.navigate(`/runs/${run.id}`),
    ...overrides,
  });

  await test('account replacement during draft cleanup never redirects or closes the replacement account workspace', async () => {
    const requests = [];
    let releaseCleanup;
    let started;
    const cleanupStarted = new Promise((resolve) => {
      started = resolve;
    });
    globalThis.window = { location: { hash: '#/runs' } };
    transport.setSessionCsrf('alice-csrf', 'alice');
    globalThis.fetch = async (path, options) => {
      requests.push({ path, options });
      if (path === '/api/v1/runs') return new Response('{"id":"alice-run"}', { status: 201 });
      started();
      return new Promise((resolve) => {
        releaseCleanup = resolve;
      });
    };
    let closes = 0;
    const pending = submitTrainingRun(
      options({
        finishDraft: () =>
          transport.api('/training-draft', {
            method: 'DELETE',
            headers: { 'X-Training-Draft-User': 'alice' },
            body: '{"expected_revision":1}',
          }),
        onSubmitted: (run) => {
          closes += 1;
          transport.navigate(`/runs/${run.id}`);
        },
      }),
    );
    await cleanupStarted;
    transport.setSessionCsrf('bob-csrf', 'bob');
    globalThis.window.location.hash = '#/assets';
    releaseCleanup(new Response(null, { status: 204 }));
    await assert.rejects(pending, (error) => error.code === 'SESSION_CHANGED');
    assert.equal(closes, 0);
    assert.equal(globalThis.window.location.hash, '#/assets');
    assert.deepEqual(
      requests.map((request) => request.options.headers.get('X-Operation-User')),
      ['alice', 'alice'],
    );
    assert.equal(requests[0].options.headers.get('Idempotency-Key'), 'run-submission-alice');
    assert.equal(JSON.parse(requests[0].options.body).confirmed_config_sha256, 'a'.repeat(64));
  });

  await test('same-account draft cleanup failures preserve navigation to an already accepted run', async () => {
    transport.setSessionCsrf('alice-current', 'alice');
    globalThis.window = { location: { hash: '#/assets' } };
    globalThis.fetch = async () => new Response('{"id":"accepted-run"}', { status: 201 });
    const run = await submitTrainingRun(
      options({
        finishDraft: async () => {
          throw new transport.ApiError('NETWORK_ERROR', 'cleanup unavailable', 0);
        },
      }),
    );
    assert.equal(run.id, 'accepted-run');
    assert.equal(globalThis.window.location.hash, '/runs/accepted-run');
  });

  await test('an explicit session-changed cleanup error is never swallowed even with the same local token', async () => {
    transport.setSessionCsrf('alice-current', 'alice');
    globalThis.window = { location: { hash: '#/assets' } };
    globalThis.fetch = async () => new Response('{"id":"accepted-run"}', { status: 201 });
    await assert.rejects(
      submitTrainingRun(
        options({
          finishDraft: async () => {
            throw new transport.ApiError('SESSION_CHANGED', 'account changed', 401);
          },
        }),
      ),
      (error) => error.code === 'SESSION_CHANGED',
    );
    assert.equal(globalThis.window.location.hash, '#/assets');
  });

  await test('closing the original wizard during cleanup stops its later navigation', async () => {
    transport.setSessionCsrf('alice-current', 'alice');
    globalThis.window = { location: { hash: '#/assets' } };
    globalThis.fetch = async () => new Response('{"id":"accepted-run"}', { status: 201 });
    let release;
    let started;
    const cleanupStarted = new Promise((resolve) => {
      started = resolve;
    });
    let active = true;
    const pending = submitTrainingRun(
      options({
        finishDraft: async () => {
          started();
          await new Promise((resolve) => {
            release = resolve;
          });
        },
        isActive: () => active,
      }),
    );
    await cleanupStarted;
    active = false;
    release();
    await assert.rejects(pending, (error) => error.name === 'AbortError');
    assert.equal(globalThis.window.location.hash, '#/assets');
  });

  await test('a wizard already closed before submission makes no API request', async () => {
    transport.setSessionCsrf('alice-current', 'alice');
    let requests = 0;
    globalThis.fetch = async () => {
      requests += 1;
      return new Response('{}');
    };
    await assert.rejects(
      submitTrainingRun(options({ isActive: () => false })),
      (error) => error.name === 'AbortError',
    );
    assert.equal(requests, 0);
  });
} finally {
  globalThis.fetch = originalFetch;
  globalThis.window = originalWindow;
  await server.close();
}
