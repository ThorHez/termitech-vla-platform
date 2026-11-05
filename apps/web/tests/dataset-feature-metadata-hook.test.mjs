import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
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
    /* Browser checks are optional when the test runner has no Playwright. */
  }
}
const executablePath =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  (process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    : '');
const available = Boolean(chromium && executablePath && existsSync(executablePath));

const annotation = (key, role, width) => ({
  key,
  role,
  representation: 'unknown',
  value_mode: 'unknown',
  component: 'other',
  arm: 'unknown',
  shape: [width],
  dtype: 'float32',
  ...(role === 'action' ? { horizon: 1 } : {}),
});
const inspection = (datasetPath, width, imageShape, alternateShape) => {
  const features = [
    { key: 'observation.state', dtype: 'float32', shape: [width], names: [] },
    { key: 'action', dtype: 'float32', shape: [width], names: [] },
    { key: 'observation.other', dtype: 'float32', shape: [11], names: [] },
    { key: 'observation.images.main', dtype: 'image', shape: imageShape, names: [] },
    { key: 'observation.images.alt', dtype: 'video', shape: alternateShape, names: [] },
  ];
  return {
    schema_version: 1,
    path: datasetPath,
    source: { info_sha256: 'a'.repeat(64), tasks_sha256: null },
    features,
    suggestions: {
      format: 'lerobot',
      feature_keys: features.map((feature) => feature.key),
      field_annotations: [
        annotation('observation.state', 'state', width),
        annotation('action', 'action', width),
      ],
    },
    processing: {},
    evidence: [],
    warnings: [],
  };
};
const query = (nodeId, datasetPath, value, id = 'inspection') => ({
  id,
  node_id: nodeId,
  path: datasetPath,
  status: 'READY',
  inspection: value,
  error: null,
});
const waitFor = async (predicate) => {
  for (let attempt = 0; attempt < 500; attempt++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.ok(predicate(), 'fixture request did not reach its expected lifecycle');
};

await test(
  'registration metadata displays only the current dataset shapes without replacing user fields',
  { skip: !available },
  async (t) => {
    const entry = `
import React,{useState}from'react';import{createRoot}from'react-dom/client';
import{AuthContext}from'/src/auth.tsx';import{setSessionCsrf}from'/src/api.ts';
import{DatasetInspection,useDatasetInspectionDraft}from'/src/DatasetInspection.tsx';
import{DatasetProfileForm,emptyDatasetProfile}from'/src/DatasetProfileForm.tsx';
import'/src/styles.css';import'/src/theme.css';
function Probe({config}){const[profile,setProfile]=useState(()=>({...emptyDatasetProfile(),robot_type:'custom robot',
  task_description:'Keep this task',cameras:[{name:'My wrist',key:'observation.images.main',kind:'rgb'}]}));
const draft=useDatasetInspectionDraft(profile,setProfile);window.__applyTemplate=value=>draft.applyTemplate(value);
const features=draft.featuresFor(config.node?.id,config.path);
return <><DatasetInspection compact node={config.node} path={config.path} value={profile} draft={draft}/>
<div id="profile-form"><DatasetProfileForm compact value={profile} onChange={draft.manualChange} onFieldEdit={draft.touch} features={features}/></div>
<pre id="fresh-features" style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere',maxWidth:'100%'}}>{JSON.stringify(features??null)}</pre>
<pre id="profile-value" style={{whiteSpace:'pre-wrap',overflowWrap:'anywhere',maxWidth:'100%'}}>{JSON.stringify(profile)}</pre></>;}
function App(){const[config,setConfig]=useState(window.__initial);window.__update=value=>setConfig(old=>({...old,...value}));
return <AuthContext.Provider value={{user:{id:'alice'},can:()=>true,loading:false,expired:false}}><Probe config={config}/></AuthContext.Provider>;}
setSessionCsrf('fixture','alice');createRoot(document.getElementById('root')).render(<App/>);`;
    const cacheDir = await mkdtemp(path.join(tmpdir(), 'vla-dataset-feature-metadata-'));
    t.after(() => rm(cacheDir, { recursive: true, force: true }));
    const server = await createServer({
      root: fileURLToPath(new URL('..', import.meta.url)),
      configFile: false,
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
          name: 'dataset-feature-metadata-fixture',
          resolveId(id) {
            if (id === '/__dataset-features__.tsx') return '\0dataset-feature-metadata.tsx';
          },
          async load(id) {
            if (id === '\0dataset-feature-metadata.tsx')
              return (
                await transformWithEsbuild(entry, 'dataset-feature-metadata.tsx', {
                  loader: 'tsx',
                  jsx: 'automatic',
                })
              ).code;
          },
          configureServer(instance) {
            instance.middlewares.use('/__fixture__', (_request, response) => {
              response.setHeader('Content-Type', 'text/html');
              response.end(
                '<!doctype html><div id="root"></div><script type="module" src="/__dataset-features__.tsx"></script>',
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
        const context = await browser.newContext({ viewport: { width: 1440, height: 1200 } });
        contexts.push(context);
        const page = await context.newPage();
        const requests = [],
          errors = [],
          loadingErrors = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => {
          if (message.type() === 'error') loadingErrors.push(message.text());
        });
        await page.addInitScript(
          (initial) => {
            window.__initial = initial;
          },
          {
            node: { id: 'node-a', status: 'ONLINE' },
            path: '/datasets/a',
            ...config,
          },
        );
        await page.route('**/*', async (route) => {
          const request = route.request(),
            url = new URL(request.url());
          assert.equal(url.origin, origin, 'fixture cannot contact a real server');
          if (!url.pathname.startsWith('/api/v1/')) return route.continue();
          assert.equal(request.method(), 'POST');
          const match = /^\/api\/v1\/nodes\/([^/]+)\/dataset-inspections$/.exec(url.pathname);
          assert.ok(match, 'fixture only allows metadata inspection requests');
          const body = request.postDataJSON();
          const current = { nodeId: decodeURIComponent(match[1]), path: body.path };
          requests.push(current);
          const response = await handler(current, requests.length);
          try {
            await route.fulfill({
              status: 200,
              contentType: 'application/json',
              body: JSON.stringify(response),
            });
          } catch {
            /* React cleanup can abort an obsolete metadata response. */
          }
        });
        await page.goto(`${origin}/__fixture__`);
        try {
          await page.locator('#fresh-features').waitFor({ timeout: 10000 });
        } catch (error) {
          throw new Error(
            `${error.message}; page errors: ${errors.join('; ')}; loading errors: ${loadingErrors.join('; ')}`,
          );
        }
        return {
          page,
          requests,
          errors,
          image: () => page.locator('.dataset-camera-shape').textContent(),
          fresh: () => page.locator('#fresh-features').textContent().then(JSON.parse),
          profile: () => page.locator('#profile-value').textContent().then(JSON.parse),
          update: (value) => page.evaluate((next) => window.__update(next), value),
          ready: () =>
            page.waitForFunction(
              () => JSON.parse(document.querySelector('#fresh-features').textContent) !== null,
            ),
          cleared: () =>
            page.waitForFunction(
              () => document.querySelector('#fresh-features').textContent === 'null',
            ),
          close: () => context.close(),
        };
      };

      await t.test(
        'node and path changes hide old shapes before replacement metadata arrives',
        async () => {
          let releaseB;
          const fixture = await mount({}, ({ nodeId, path: selectedPath }) =>
            selectedPath === '/datasets/a'
              ? query(
                  nodeId,
                  selectedPath,
                  inspection(selectedPath, 7, [480, 640, 3], [3, 120, 160]),
                )
              : new Promise((resolve) => {
                  releaseB = () =>
                    resolve(
                      query(
                        nodeId,
                        selectedPath,
                        inspection(selectedPath, 14, [3, 224, 320], [64, 96, 3]),
                      ),
                    );
                }),
          );
          await fixture.ready();
          assert.match(await fixture.image(), /\[480, 640, 3\]/);
          await fixture.update({ node: { id: 'node-b', status: 'ONLINE' }, path: '/datasets/b' });
          await fixture.cleared();
          assert.doesNotMatch(await fixture.image(), /480|640/);
          assert.equal((await fixture.profile()).robot_type, 'custom robot');
          assert.equal((await fixture.profile()).cameras[0].name, 'My wrist');
          assert.equal((await fixture.profile()).cameras[0].key, 'observation.images.main');
          assert.equal((await fixture.profile()).task_description, 'Keep this task');
          await waitFor(() => releaseB);
          releaseB();
          await fixture.ready();
          assert.match(await fixture.image(), /\[3, 224, 320\]/);
          assert.match(await fixture.image(), /宽 320 × 高 224/);
          if (process.env.VLA_DATASET_SHAPE_SCREENSHOT) {
            await mkdir(path.dirname(process.env.VLA_DATASET_SHAPE_SCREENSHOT), {
              recursive: true,
            });
            await fixture.page
              .locator('fieldset.dataset-profile-form')
              .screenshot({ path: process.env.VLA_DATASET_SHAPE_SCREENSHOT });
          }
          assert.deepEqual(fixture.requests, [
            { nodeId: 'node-a', path: '/datasets/a' },
            { nodeId: 'node-b', path: '/datasets/b' },
          ]);
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test(
        'a failed reread at the same location clears fresh image facts without losing input',
        async () => {
          let releaseFailure;
          const fixture = await mount({}, ({ nodeId, path: selectedPath }, number) =>
            number === 1
              ? query(
                  nodeId,
                  selectedPath,
                  inspection(selectedPath, 7, [480, 640, 3], [3, 120, 160]),
                )
              : new Promise((resolve) => {
                  releaseFailure = () =>
                    resolve({
                      ...query(nodeId, selectedPath, null),
                      status: 'FAILED',
                      error: { code: 'FIXTURE_FAILED', message: 'fixture metadata unavailable' },
                    });
                }),
          );
          await fixture.ready();
          await fixture.page
            .getByRole('button', { name: '读取数据集字段，推断并填入默认值', exact: true })
            .click();
          await fixture.cleared();
          assert.doesNotMatch(await fixture.image(), /480|640/);
          await waitFor(() => releaseFailure);
          releaseFailure();
          await fixture.page.getByText('fixture metadata unavailable', { exact: true }).waitFor();
          assert.equal(await fixture.fresh(), null);
          assert.match(await fixture.image(), /待读取数据集元数据/);
          assert.equal((await fixture.profile()).cameras[0].key, 'observation.images.main');
          assert.deepEqual(fixture.errors, []);
          await fixture.close();
        },
      );

      await t.test(
        'template-only facts never invent image shape and field edits resolve fresh metadata by key',
        async () => {
          const fixture = await mount(
            { node: { id: 'node-a', status: 'OFFLINE' } },
            ({ nodeId, path: selectedPath }) =>
              query(
                nodeId,
                selectedPath,
                inspection(selectedPath, 7, [480, 640, 3], [3, 120, 160]),
              ),
          );
          const template = await fixture.profile();
          template.field_annotations = [annotation('observation.state', 'state', 99)];
          await fixture.page.evaluate((value) => window.__applyTemplate(value), template);
          assert.equal(await fixture.fresh(), null);
          assert.match(await fixture.image(), /待读取数据集元数据/);
          await fixture.update({ node: { id: 'node-a', status: 'ONLINE' } });
          await fixture.ready();
          assert.match(await fixture.image(), /\[480, 640, 3\]/);
          // The source metadata has width seven even though the template declared 99.
          assert.match(
            await fixture.page.getByLabel('字段原始信息 1', { exact: true }).textContent(),
            /状态维度：7 维/,
          );
          await fixture.page
            .getByPlaceholder('例如 observation.images.left_wrist', { exact: true })
            .fill('observation.images.alt');
          assert.match(await fixture.image(), /\[3, 120, 160\]/);
          assert.match(await fixture.image(), /宽 160 × 高 120/);
          const sourceField = fixture.page.getByLabel('标签真实字段 1', { exact: true });
          if (await sourceField.evaluate((element) => element.tagName === 'SELECT'))
            await sourceField.selectOption('observation.other');
          else await sourceField.fill('observation.other');
          assert.match(
            await fixture.page.getByLabel('字段原始信息 1', { exact: true }).textContent(),
            /状态维度：11 维/,
          );
          assert.equal((await fixture.profile()).field_annotations[0].shape, undefined);
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
