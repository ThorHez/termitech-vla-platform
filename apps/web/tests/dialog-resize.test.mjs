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
  const { fitDialogBounds, resizeDialogBounds } = await server.ssrLoadModule('/src/dialogResizeBounds.ts');
  const initial = { left: 200, top: 150, width: 600, height: 400 };
  const viewport = { width: 1200, height: 800 };

  await test('each border and corner changes its own edges while retaining the opposite edges', () => {
    for (const edge of ['n', 'e', 's', 'w', 'ne', 'nw', 'se', 'sw']) {
      const changed = resizeDialogBounds(initial, edge, 30, 20, viewport);
      assert.equal(changed.left, edge.includes('w') ? 230 : 200, edge);
      assert.equal(changed.top, edge.includes('n') ? 170 : 150, edge);
      assert.equal(changed.left + changed.width, edge.includes('e') ? 830 : 800, edge);
      assert.equal(changed.top + changed.height, edge.includes('s') ? 570 : 550, edge);
    }
  });

  await test('overshooting a border cannot invert the dialog or move it outside the viewport', () => {
    for (const edge of ['n', 'e', 's', 'w', 'ne', 'nw', 'se', 'sw']) {
      for (const delta of [-100000, 100000]) {
        const value = resizeDialogBounds(initial, edge, delta, delta, viewport);
        assert.ok(value.width >= 360 && value.height >= 240);
        assert.ok(value.left >= 12 && value.top >= 12);
        assert.ok(value.left + value.width <= 1188);
        assert.ok(value.top + value.height <= 788);
      }
    }
  });

  await test('a previously resized desktop dialog fits a phone or short viewport', () => {
    for (const viewport of [{ width: 390, height: 844 }, { width: 800, height: 200 }, { width: 280, height: 180 }]) {
      const result = fitDialogBounds({ left: 900, top: 500, width: 1000, height: 700 }, viewport);
      assert.ok(result.left >= 12 && result.top >= 12);
      assert.ok(result.left + result.width <= viewport.width - 12);
      assert.ok(result.top + result.height <= viewport.height - 12);
      assert.ok(result.width > 0 && result.height > 0);
    }
  });

  await test('horizontal resizing preserves the initial height of a small notice', () => {
    const small = { left: 200, top: 150, width: 500, height: 160 };
    assert.equal(resizeDialogBounds(small, 'e', 50, 0, viewport).height, 160);
  });

  await test('the shared modal supplies eight non-submitting handles and an accessible keyboard grip', async () => {
    const { Modal } = await server.ssrLoadModule('/src/components.tsx');
    const html = renderToStaticMarkup(React.createElement(Modal, {
      title: '登记本地资产', close() {}, footer: React.createElement('button', null, '登记资产'),
    }, React.createElement('input', { defaultValue: 'preserved' })));
    assert.equal((html.match(/class="dialog-resize-handle /g) || []).length, 8);
    assert.equal((html.match(/type="button" class="dialog-resize-handle /g) || []).length, 8);
    assert.match(html, /aria-label="调整弹窗大小/);
    assert.match(html, /dialog-resize-se[^>]*tabindex="0"/);
    assert.match(html, /role="dialog"[^>]*aria-modal="true"/);
    assert.match(html, /value="preserved"/);
  });
} finally {
  await server.close();
}
