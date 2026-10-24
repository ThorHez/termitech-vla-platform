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
  const { AuthContext } = await server.ssrLoadModule('/src/auth.tsx');
  const { DataProvider } = await server.ssrLoadModule('/src/data.tsx');
  const { NodeRename, nodeNameError } = await server.ssrLoadModule('/src/NodeRename.tsx');
  const { NodePage } = await server.ssrLoadModule('/src/pages.tsx');
  const render = (permissions, component, props, withData = true) => {
    const element = React.createElement(component, props);
    return renderToStaticMarkup(
      React.createElement(
        AuthContext.Provider,
        { value: { can: (permission) => permissions.includes(permission) } },
        withData ? React.createElement(DataProvider, { enabled: true }, element) : element,
      ),
    );
  };

  await test('rename dialog starts with the current name and a disabled unchanged save', () => {
    const html = render(['machines.manage'], NodeRename, {
      node: { id: 'machine', name: '训练机 A <prod>' },
      close() {},
    });
    assert.match(html, /role="dialog"[^>]*aria-modal="true"[^>]*aria-label="重命名机器"/);
    assert.match(html, /name="name"[^>]*value="训练机 A &lt;prod&gt;"/);
    assert.match(html, /<button[^>]*type="submit"[^>]*disabled=""[^>]*>保存名称<\/button>/);
    assert.match(html, /用于机器列表、训练任务和数据资产中的机器名称/);
    assert.match(html, />取消<\/button>/);
  });

  await test('only machine administrators can reach rename controls', () => {
    const props = { node: { id: 'machine', name: '机器' }, close() {} };
    for (const permissions of [[], ['platform.read'], ['training.write', 'assets.write']]) {
      const dialog = render(permissions, NodeRename, props, false);
      assert.match(dialog, /重命名机器需要管理员权限/);
      assert.doesNotMatch(dialog, /name="name"|保存名称/);
      const detail = render(permissions, NodePage, { id: 'machine', createRun() {} });
      assert.doesNotMatch(detail, />重命名<\/button>/);
    }
    assert.match(
      render(['machines.manage'], NodePage, { id: 'machine', createRun() {} }),
      />重命名<\/button>/,
    );
  });

  await test('names accept Chinese and Unicode while excluding empty, overlong and control text', () => {
    assert.equal(nodeNameError('  中文训练机  '), undefined);
    assert.equal(nodeNameError('🦾'.repeat(200)), undefined);
    assert.match(nodeNameError('🦾'.repeat(201)), /200/);
    assert.match(nodeNameError(' \t\n '), /请输入/);
    for (const value of ['机器\u0000', '机器\nA', '机器\u007f', '机器\u0085']) {
      assert.match(nodeNameError(value), /控制字符/);
    }
  });
} finally {
  await server.close();
}
