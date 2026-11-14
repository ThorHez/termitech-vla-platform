import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { homedir, tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

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
    // Browser checks are optional where Playwright is unavailable.
  }
}
const executablePath =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  (process.platform === 'darwin'
    ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
    : '');
const available = Boolean(chromium && executablePath && existsSync(executablePath));
const timestamp = (seconds) => `2026-11-14T01:02:${seconds}.000000+00:00`;

await test(
  'asset deletion requires an exact path and current inspection and renders Agent status',
  { skip: !available },
  async (t) => {
    const cacheDir = await mkdtemp(path.join(tmpdir(), 'vla-asset-deletions-ui-'));
    const server = await createServer({
      root: fileURLToPath(new URL('..', import.meta.url)),
      cacheDir,
      server: { host: '127.0.0.1', port: 0, hmr: false, proxy: {} },
    });
    t.after(async () => {
      await server.close();
      await rm(cacheDir, { recursive: true, force: true });
    });
    await server.listen();
    const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
    const browser = await chromium.launch({ executablePath, headless: true });
    t.after(() => browser.close());

    const asset = {
      id: 'fixture-asset',
      name: 'Archived dataset',
      kind: 'DATASET',
      node_id: 'fixture-node',
      node_name: 'Fixture node',
      path: '/fixture/storage/with space/dataset',
      can_purge: true,
      inspection: null,
      file_deletion: null,
      archived_at: null,
      archived_by: null,
    };
    const scope = `/api/v1/assets/${asset.id}`;
    const commands = [];
    const page = await browser.newPage();
    let role = 'admin';
    let reauthenticated = false;
    let ledgerReads = 0;
    const session = () => ({
      user: {
        id: role,
        username: role,
        display_name: role,
        role,
        enabled: true,
        must_change_password: false,
      },
      permissions: role === 'admin' ? ['assets.purge'] : [],
      csrf_token: 'fixture',
      server_time: new Date().toISOString(),
      expires_at: new Date(Date.now() + 3600000).toISOString(),
      idle_expires_at: new Date(Date.now() + 3600000).toISOString(),
      session_idle_seconds: 3600,
      reauth_expires_at: reauthenticated ? new Date(Date.now() + 300000).toISOString() : null,
    });
    const inspection = (id, status, seconds) => ({
      id,
      status,
      requested_at: timestamp(seconds),
      error_code: null,
      error_message: null,
    });
    const deletion = (id, status, seconds, extra = {}) => ({
      id,
      status,
      requested_at: timestamp(seconds),
      finished_at: null,
      error_code: null,
      error_message: null,
      ...extra,
    });
    await page.route('**/*', async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      assert.equal(url.origin, origin, 'the fixture must stay on localhost');
      const endpoint = url.pathname;
      if (!endpoint.startsWith('/api/v1/')) return route.continue();
      const respond = (json, status = 200) => route.fulfill({ json, status });
      if (endpoint === '/api/v1/auth/me' || endpoint === '/api/v1/auth/activity')
        return respond(session());
      if (endpoint === '/api/v1/auth/reauth') {
        assert.deepEqual(request.postDataJSON(), { password: 'fixture-only-password' });
        reauthenticated = true;
        return respond(session());
      }
      if (endpoint === '/api/v1/assets/deletions') {
        ledgerReads++;
        return respond([asset]);
      }
      if (endpoint === `${scope}/purge-inspections`) {
        assert.equal(request.method(), 'POST');
        commands.push('inspect');
        asset.inspection =
          commands.length === 1
            ? inspection('first', 'RUNNING', '10')
            : inspection('second', 'PENDING', '30');
        return respond(asset.inspection, 202);
      }
      if (endpoint === `${scope}/purge`) {
        assert.equal(request.method(), 'POST');
        if (!reauthenticated)
          return respond({ error: { code: 'REAUTH_REQUIRED', message: 'Password required' } }, 403);
        commands.push(request.postDataJSON());
        asset.file_deletion = deletion(`purge-${commands.length}`, 'PENDING', '40');
        asset.can_purge = false;
        return respond(asset.file_deletion, 202);
      }
      if (endpoint === '/api/v1/overview') return respond(null);
      return respond([]);
    });

    await page.goto(`${origin}/#/assets`);
    await page.getByRole('link', { name: '已删除资产' }).click();
    await page.getByText(asset.path, { exact: true }).waitFor();
    await page.getByText('时间未记录', { exact: true }).waitFor();
    await page.getByRole('button', { name: `永久删除文件 ${asset.name}` }).click();
    const dialog = page.getByRole('dialog', { name: '永久删除文件' });
    const field = dialog.getByRole('textbox', { name: '输入完整原文件路径' });
    const confirm = dialog.getByRole('button', { name: '确认永久删除', exact: true });
    await dialog.getByText('正在检查文件', { exact: true }).waitFor();
    assert.equal(await confirm.isDisabled(), true);
    assert.deepEqual(commands, ['inspect']);
    asset.inspection = inspection('first', 'READY', '10');
    await dialog.getByText('已核对文件路径', { exact: true }).waitFor();
    await field.fill(`${asset.path}/`);
    assert.equal(await confirm.isDisabled(), true);
    await field.fill(asset.path);
    await confirm.click();
    const password = page.getByRole('dialog', { name: '验证当前账号密码' });
    assert.deepEqual(commands, ['inspect']);
    await password.getByLabel('当前密码').fill('fixture-only-password');
    await password.getByRole('button', { name: '验证并继续' }).click();
    await page.getByText('等待删除', { exact: true }).waitFor();
    assert.deepEqual(commands[1], { confirm_path: asset.path, inspection_id: 'first' });

    asset.can_purge = true;
    asset.requires_new_inspection = true;
    asset.file_deletion = deletion('purge-2', 'FAILED', '20', {
      error_message: '原路径文件发生变化',
    });
    await page.getByText('原路径文件发生变化', { exact: true }).waitFor();
    await page.getByRole('button', { name: `重试删除文件 ${asset.name}` }).click();
    const retry = page.getByRole('dialog', { name: '重试永久删除文件' });
    await retry.getByText('文件自上次检查后已变化，请重新检查文件。').waitFor();
    assert.equal(await retry.getByRole('button', { name: '确认永久删除' }).isDisabled(), true);
    await retry.getByRole('button', { name: '重新检查文件' }).click();
    await retry.getByText('等待文件检查', { exact: true }).waitFor();
    asset.inspection = inspection('second', 'MISSING', '30');
    const missing = retry.getByRole('button', { name: '确认记录文件不存在' });
    await missing.waitFor();
    assert.equal(await missing.isDisabled(), true);
    await retry.getByRole('textbox').fill(asset.path);
    await missing.click();
    await page.getByText('等待删除', { exact: true }).waitFor();
    assert.deepEqual(commands[3], { confirm_path: asset.path, inspection_id: 'second' });
    asset.file_deletion = deletion('purge-4', 'MISSING', '40', { finished_at: timestamp('50') });
    await page.getByText('文件已不存在', { exact: true }).waitFor();

    const readsBeforeDeniedNavigation = ledgerReads;
    role = 'operator';
    await page.reload();
    await page.getByRole('heading', { name: '权限不足' }).waitFor();
    assert.equal(ledgerReads, readsBeforeDeniedNavigation);
  },
);
