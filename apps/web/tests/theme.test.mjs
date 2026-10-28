import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createServer } from 'vite';

const bootstrap = await readFile(new URL('../public/theme-init.js', import.meta.url), 'utf8');

function initialAppearance({
  saved = null,
  systemDark = false,
  storageThrows = false,
  media = true,
}) {
  const document = { documentElement: { dataset: {}, style: {} } };
  const window = media ? { matchMedia: () => ({ matches: systemDark }) } : {};
  const localStorage = {
    getItem() {
      if (storageThrows) throw new Error('Storage is blocked');
      return saved;
    },
  };
  runInNewContext(bootstrap, { document, window, localStorage });
  return document.documentElement;
}

for (const { name, expected, ...settings } of [
  { name: 'first visit follows a dark system', systemDark: true, expected: 'dark' },
  { name: 'first visit follows a light system', systemDark: false, expected: 'light' },
  {
    name: 'saved light overrides a dark system',
    saved: 'light',
    systemDark: true,
    expected: 'light',
  },
  {
    name: 'saved dark overrides a light system',
    saved: 'dark',
    systemDark: false,
    expected: 'dark',
  },
  {
    name: 'saved system follows system appearance',
    saved: 'system',
    systemDark: true,
    expected: 'dark',
  },
  {
    name: 'invalid saved preference follows system',
    saved: 'invalid',
    systemDark: true,
    expected: 'dark',
  },
  {
    name: 'blocked storage still follows system',
    storageThrows: true,
    systemDark: true,
    expected: 'dark',
  },
  { name: 'missing matchMedia falls back to light', media: false, expected: 'light' },
  { name: 'saved dark works without matchMedia', saved: 'dark', media: false, expected: 'dark' },
]) {
  await test(`theme bootstrap: ${name} before the application mounts`, () => {
    const root = initialAppearance(settings);
    assert.equal(root.dataset.theme, expected);
    assert.equal(root.style.colorScheme, expected, 'native controls match the initial page theme');
  });
}

const server = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: 'custom',
});
const originalWindow = globalThis.window;

try {
  const { ThemeProvider, ThemeControl, ThemeSettings } =
    await server.ssrLoadModule('/src/theme.tsx');
  const render = (component, saved, systemDark = false) => {
    globalThis.window = {
      localStorage: { getItem: () => saved },
      matchMedia: () => ({ matches: systemDark }),
    };
    return renderToStaticMarkup(
      React.createElement(ThemeProvider, null, React.createElement(component)),
    );
  };

  await test('theme control exposes all three choices through a labelled native select', () => {
    const html = render(ThemeControl, null);
    assert.match(html, /<select\b[^>]*aria-label="界面主题"/);
    const options = [...html.matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g)];
    assert.deepEqual(
      options.map(([, attributes, label]) => ({
        value: attributes.match(/value="([^"]*)"/)[1],
        label,
        selected: attributes.includes('selected=""'),
      })),
      [
        { value: 'light', label: '浅色模式', selected: false },
        { value: 'dark', label: '夜间模式', selected: false },
        { value: 'system', label: '跟随系统', selected: true },
      ],
    );
  });

  await test('theme control retains explicit selection independently of current system appearance', () => {
    for (const [saved, systemDark] of [
      ['light', true],
      ['dark', false],
    ]) {
      const html = render(ThemeControl, saved, systemDark);
      const selected = [...html.matchAll(/<option\b([^>]*)>/g)]
        .map(([, attributes]) => attributes)
        .filter((attributes) => attributes.includes('selected=""'));
      assert.equal(selected.length, 1);
      assert.match(selected[0], new RegExp(`value="${saved}"`));
    }
  });

  await test('appearance settings announce exactly the saved choice as pressed', () => {
    const labels = { light: '浅色模式', dark: '夜间模式', system: '跟随系统' };
    for (const [preference, label] of Object.entries(labels)) {
      const html = render(ThemeSettings, preference);
      assert.match(html, /role="group" aria-label="外观主题"/);
      const buttons = [...html.matchAll(/<button\b([^>]*)>([\s\S]*?)<\/button>/g)];
      assert.equal(buttons.length, 3);
      const selected = buttons.filter(([, attributes]) =>
        attributes.includes('aria-pressed="true"'),
      );
      assert.equal(selected.length, 1);
      assert.equal(selected[0][2].replace(/<[^>]*>/g, '').trim(), label);
      assert.equal(
        buttons.filter(([, attributes]) => attributes.includes('aria-pressed="false"')).length,
        2,
      );
    }
  });

  await test('invalid persisted preference renders the system choice safely', () => {
    const html = render(ThemeControl, 'unexpected', true);
    assert.match(
      html,
      /<option\b(?=[^>]*value="system")(?=[^>]*selected="")[^>]*>跟随系统<\/option>/,
    );
  });
} finally {
  if (originalWindow === undefined) delete globalThis.window;
  else globalThis.window = originalWindow;
  await server.close();
}
