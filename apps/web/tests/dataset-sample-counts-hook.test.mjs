import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { homedir, tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer, transformWithEsbuild } from 'vite';
import react from '@vitejs/plugin-react';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  try {
    ({ chromium } = require(
      path.join(
        homedir(),
        '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright',
      ),
    ));
  } catch {
    /* Browser lifecycle checks are optional where Playwright is unavailable. */
  }
}
const executablePath =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  (process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    : '');
const available = Boolean(chromium && executablePath && existsSync(executablePath));
const asset = (id, changes = {}) => ({
  id,
  kind: 'DATASET',
  node_id: 'node',
  name: id,
  path: `/data/${id}`,
  version: 'v1',
  state: 'REGISTERED',
  ...changes,
});
const bindings = (...ids) =>
  ids.map((id) => ({ dataset_asset_id: id, norm_asset_id: '', weight: 1 }));
const report = (selected, count = 100) => ({
  asset_id: selected.id,
  node_id: selected.node_id,
  checked_at: '2026-11-04T12:00:00Z',
  status: 'PARTIAL',
  dataset: { status: 'READY', path: `${selected.path}/meta/info.json`, total_frames: count },
  normalization: { status: 'FAILED' },
});

await test(
  'dataset sample count hook uses isolated live reads and preserves current selection',
  { skip: !available },
  async (t) => {
    const entry = `
import React,{useState}from'react';import{createRoot}from'react-dom/client';
import{AuthContext}from'/src/auth.tsx';import{setSessionCsrf}from'/src/api.ts';
import{useDatasetSampleCounts}from'/src/useDatasetSampleCounts.ts';
function Probe({config}){const result=useDatasetSampleCounts(config.bindings,config.assets,config.node);
window.__refresh=result.refresh;return <pre id="counts">{JSON.stringify(result.counts)}</pre>;}
function App(){const[config,setConfig]=useState(window.__initial);
window.__update=patch=>{if(Object.hasOwn(patch,'userId'))setSessionCsrf('fixture',patch.userId||undefined);setConfig(old=>({...old,...patch}));};
return <AuthContext.Provider value={{user:config.userId?{id:config.userId}:undefined,loading:false,expired:false}}><Probe config={config}/></AuthContext.Provider>;}
setSessionCsrf('fixture',window.__initial.userId);createRoot(document.getElementById('root')).render(<App/>);`;
    const cacheDir = await mkdtemp(path.join(tmpdir(), 'vla-sample-counts-hook-'));
    t.after(() => rm(cacheDir, { recursive: true, force: true }));
    const server = await createServer({
      root: fileURLToPath(new URL('..', import.meta.url)),
      configFile: false,
      // Other test workers use different Vite configs; never replace their optimizer cache.
      cacheDir,
      optimizeDeps: {
        noDiscovery: true,
        include: [
          'react',
          'react-dom/client',
          'react/jsx-runtime',
          'react/jsx-dev-runtime',
          'lucide-react',
        ],
      },
      resolve: { dedupe: ['react', 'react-dom'] },
      plugins: [
        react(),
        {
          name: 'sample-counts-hook-fixture',
          resolveId(id) {
            if (id === '/__sample-counts__.tsx') return '\0sample-counts-hook.tsx';
          },
          async load(id) {
            if (id === '\0sample-counts-hook.tsx')
              return (
                await transformWithEsbuild(entry, 'sample-counts-hook.tsx', {
                  loader: 'tsx',
                  jsx: 'automatic',
                })
              ).code;
          },
          configureServer(instance) {
            instance.middlewares.use('/__fixture__', (_request, response) => {
              response.setHeader('Content-Type', 'text/html');
              response.end(
                '<!doctype html><div id="root"></div><script type="module" src="/__sample-counts__.tsx"></script>',
              );
            });
          },
        },
      ],
      server: { host: '127.0.0.1', port: 0, hmr: false },
    });
    let browser;
    const contexts = [];
    try {
      await server.listen();
      const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
      browser = await chromium.launch({ executablePath, headless: true });
      const mount = async (config, handler) => {
        const context = await browser.newContext();
        contexts.push(context);
        const page = await context.newPage();
        const requests = [],
          errors = [],
          loadingErrors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => {
          if (message.type() === 'error') loadingErrors.push(message.text());
        });
        page.on('requestfailed', (request) => {
          loadingErrors.push(`${new URL(request.url()).pathname}: ${request.failure()?.errorText}`);
        });
        page.on('response', (response) => {
          if (response.status() >= 400)
            loadingErrors.push(`${new URL(response.url()).pathname}: HTTP ${response.status()}`);
        });
        await page.addInitScript(
          (initial) => {
            window.__initial = initial;
          },
          {
            userId: 'alice',
            node: { id: 'node', status: 'ONLINE' },
            ...config,
          },
        );
        await page.route('**/*', async (route) => {
          const request = route.request(),
            url = new URL(request.url());
          assert.equal(url.origin, origin, 'fixture cannot contact a real server');
          if (!url.pathname.startsWith('/api/v1/')) return route.continue();
          assert.equal(request.method(), 'GET');
          const match = /^\/api\/v1\/assets\/([^/]+)\/statistics$/.exec(url.pathname);
          assert.ok(match, 'only dataset statistics reads are allowed');
          const id = decodeURIComponent(match[1]);
          requests.push(id);
          const body = await handler(id, requests.length);
          try {
            await route.fulfill({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify(body),
            });
          } catch {
            /* An obsolete read may already have been aborted by React cleanup. */
          }
        });
        await page.goto(`${origin}/__fixture__`);
        try {
          await page.locator('#counts').waitFor({ timeout: 10000 });
        } catch (error) {
          throw new Error(
            `${error.message}; page errors: ${errors.join('; ')}; loading errors: ${loadingErrors.join('; ')}`,
          );
        }
        const wait = (predicate) => page.waitForFunction(predicate);
        const counts = () => page.locator('#counts').textContent().then(JSON.parse);
        const update = (patch) => page.evaluate((value) => window.__update(value), patch);
        return { page, requests, errors, wait, counts, update, close: () => context.close() };
      };

      await t.test(
        'successful validation avoids SSH; partial normalization and real zero frames remain usable',
        async () => {
          const a = asset('a', {
            validation: {
              status: 'SUCCEEDED',
              checked_at: '2026-11-04T11:00:00Z',
              timestamp_contract: { status: 'verified', checked_rows: 120 },
            },
          });
          const b = asset('b', {
            validation: { status: 'STALE', timestamp_contract: { checked_rows: 999 } },
          });
          const fixture = await mount({ assets: [a, b], bindings: bindings('a', 'a', 'b') }, (id) =>
            report(id === 'a' ? a : b, id === 'a' ? 200 : 0),
          );
          await fixture.wait(
            () => JSON.parse(document.querySelector('#counts').textContent).b?.status === 'ready',
          );
          assert.deepEqual(fixture.requests, ['b']);
          assert.equal((await fixture.counts()).a.count, 120);
          assert.equal((await fixture.counts()).a.source, 'validation');
          assert.equal((await fixture.counts()).b.count, 0);
          await fixture.page.evaluate(() => window.__refresh());
          await fixture.wait(
            () => JSON.parse(document.querySelector('#counts').textContent).a?.count === 200,
          );
          assert.deepEqual([...fixture.requests].sort(), ['a', 'b', 'b']);
          assert.equal((await fixture.counts()).a.source, 'metadata');
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test(
        'adding a dataset reuses unchanged successful counts; failed refresh cannot resurrect old success',
        async () => {
          const selected = [asset('a'), asset('b')];
          let fail = false;
          const fixture = await mount({ assets: selected, bindings: bindings('a') }, (id) => {
            const value = report(
              selected.find((item) => item.id === id),
              id === 'a' ? 100 : 200,
            );
            return fail ? { ...value, dataset: { ...value.dataset, status: 'FAILED' } } : value;
          });
          await fixture.wait(
            () => JSON.parse(document.querySelector('#counts').textContent).a?.status === 'ready',
          );
          await fixture.update({ bindings: bindings('a', 'b') });
          await fixture.wait(
            () => JSON.parse(document.querySelector('#counts').textContent).b?.status === 'ready',
          );
          assert.deepEqual(fixture.requests, ['a', 'b']);
          fail = true;
          await fixture.page.evaluate(() => window.__refresh());
          await fixture.wait(() =>
            Object.values(JSON.parse(document.querySelector('#counts').textContent)).every(
              (value) => value.status === 'unavailable',
            ),
          );
          assert.equal((await fixture.counts()).a.count, undefined);
          assert.deepEqual(fixture.requests, ['a', 'b', 'a', 'b']);
          fail = false;
          await fixture.page.evaluate(() => window.__refresh());
          await fixture.wait(() =>
            Object.values(JSON.parse(document.querySelector('#counts').textContent)).every(
              (value) => value.status === 'ready',
            ),
          );
          assert.equal((await fixture.counts()).a.count, 100);
          await fixture.close();
        },
      );

      await t.test(
        'missing and offline sources never issue statistics reads, and logout hides existing counts',
        async () => {
          const a = asset('a');
          const fixture = await mount(
            {
              assets: [a],
              bindings: bindings('a', 'missing'),
              node: { id: 'node', status: 'OFFLINE' },
            },
            () => report(a),
          );
          await fixture.wait(
            () =>
              JSON.parse(document.querySelector('#counts').textContent).a?.status === 'unavailable',
          );
          assert.deepEqual(fixture.requests, []);
          await fixture.update({ node: { id: 'node', status: 'ONLINE' } });
          await fixture.wait(
            () => JSON.parse(document.querySelector('#counts').textContent).a?.status === 'ready',
          );
          assert.deepEqual(fixture.requests, ['a']);
          assert.equal((await fixture.counts()).missing.status, 'unavailable');
          await fixture.update({ userId: null });
          await fixture.wait(
            () =>
              JSON.parse(document.querySelector('#counts').textContent).a?.status === 'unavailable',
          );
          assert.equal((await fixture.counts()).a.count, undefined);
          await fixture.page.evaluate(() => window.__refresh());
          assert.deepEqual(fixture.requests, ['a']);
          await fixture.close();
        },
      );

      await t.test('a two-read queue deduplicates repeated dataset IDs', async () => {
        const selected = ['a', 'b', 'c'].map((id) => asset(id));
        const releases = new Map();
        const fixture = await mount(
          { assets: selected, bindings: bindings('a', 'a', 'b', 'c') },
          (id) =>
            new Promise((resolve) =>
              releases.set(id, () => resolve(report(selected.find((item) => item.id === id)))),
            ),
        );
        await fixture.page.waitForFunction(
          () => Object.keys(JSON.parse(document.querySelector('#counts').textContent)).length === 3,
        );
        for (let attempt = 0; fixture.requests.length < 2 && attempt < 100; attempt++)
          await new Promise((resolve) => setTimeout(resolve, 10));
        assert.deepEqual(fixture.requests, ['a', 'b']);
        releases.get('a')();
        for (let attempt = 0; fixture.requests.length < 3 && attempt < 100; attempt++)
          await new Promise((resolve) => setTimeout(resolve, 10));
        assert.deepEqual(fixture.requests, ['a', 'b', 'c']);
        releases.get('b')();
        releases.get('c')();
        await fixture.wait(() =>
          Object.values(JSON.parse(document.querySelector('#counts').textContent)).every(
            (value) => value.status === 'ready',
          ),
        );
        assert.deepEqual(fixture.errors, []);
        await fixture.close();
      });

      await t.test(
        'obsolete path/account replies cannot replace current results, and weight edits do not fetch again',
        async () => {
          const old = asset('a'),
            fresh = asset('a', { path: '/data/replacement' });
          let releaseOld;
          const fixture = await mount({ assets: [old], bindings: bindings('a') }, (_id, number) =>
            number === 1
              ? new Promise((resolve) => {
                  releaseOld = () => resolve(report(old, 111));
                })
              : report(fresh, 222),
          );
          for (let attempt = 0; !releaseOld && attempt < 100; attempt++)
            await new Promise((resolve) => setTimeout(resolve, 10));
          assert.ok(releaseOld);
          await fixture.update({ assets: [fresh], userId: 'bob' });
          await fixture.wait(
            () => JSON.parse(document.querySelector('#counts').textContent).a?.count === 222,
          );
          releaseOld();
          await fixture.update({ bindings: [{ ...bindings('a')[0], weight: 99 }] });
          await fixture.page.waitForTimeout(100);
          assert.equal((await fixture.counts()).a.count, 222);
          assert.deepEqual(fixture.requests, ['a', 'a']);
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test(
        'wrong identity, path, readiness and invalid frame counts stay unavailable',
        async () => {
          const invalid = [
            { asset_id: 'different' },
            { node_id: 'different' },
            { dataset: { status: 'READY', path: '/other/path', total_frames: 10 } },
            ...[-1, 1.5, '100', Number.MAX_SAFE_INTEGER + 1].map((count) => ({
              dataset: { status: 'READY', total_frames: count },
            })),
            { dataset: { status: 'FAILED', total_frames: 10 } },
          ];
          const selected = invalid.map((_value, index) =>
            asset(`invalid-${index}`, {
              validation: {
                status: 'SUCCEEDED',
                timestamp_contract: { status: 'pending', checked_rows: 999 },
              },
            }),
          );
          const fixture = await mount(
            { assets: selected, bindings: bindings(...selected.map((item) => item.id)) },
            (id) => {
              const current = selected.find((item) => item.id === id),
                change = invalid[selected.indexOf(current)];
              const value = report(current);
              return { ...value, ...change, dataset: { ...value.dataset, ...change.dataset } };
            },
          );
          await fixture.wait(() =>
            Object.values(JSON.parse(document.querySelector('#counts').textContent)).every(
              (value) => value.status === 'unavailable',
            ),
          );
          assert.ok(
            Object.values(await fixture.counts()).every((value) => value.count === undefined),
          );
          assert.equal(fixture.requests.length, selected.length);
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );
    } finally {
      for (const context of contexts) await context.close().catch(() => {});
      await browser?.close();
      await server.close();
    }
  },
);
