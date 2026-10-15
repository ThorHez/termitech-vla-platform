# VLA 管理前端

React、TypeScript 和 Vite 实现的中文管理台。默认读取真实管理 API，不包含默认演示数据。

## 本地开发

使用 Node.js 22 和 `package.json` 固定的 pnpm 版本：

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

打开 `http://127.0.0.1:5173`。在「平台设置」中填写管理服务的 `VLA_ADMIN_TOKEN`，令牌保存在当前标签页的 `sessionStorage`。开发服务器把 `/api` 和 `/healthz` 代理到 `http://127.0.0.1:8000`。

```bash
pnpm build
pnpm format:check
```

`build` 会先执行严格 TypeScript 检查，再输出 `dist/`。生产环境需要静态文件服务器和同源 `/api` 反向代理。`vite preview` 仅用于检查静态构建，不提供开发 API 代理。

## 功能范围

- 工作台：真实资源汇总、近期任务、当前异常。
- 机器：GPU 实测指标、平台预约、外部进程、本机资产、运行环境。
- 资产：按机器和类型筛选、本机路径登记、逻辑归档。
- 训练：模板、同机输入选择、参数、异步预检、完整代码快照、幂等提交、取消。
- 详情：实际训练指标、增量日志、进程资源、实际代码版本、冻结配置、输入与产出关联。

MEM、ReCap、RLT 等 `PLANNED` 模板展示为待验收并禁止提交。`smoke` 模板明确标识为控制链路测试。界面按模板定义渲染参数，只允许选择 Agent 登记且与模板匹配的本机环境。

## 数据更新

资源列表每 8 秒刷新，训练详情和事件每 3 秒刷新，预检每 1.8 秒刷新。日志和事件通过 `after_seq` 继续读取，每次最多 1,000 条；页面保留最近 5,000 条，导出按钮导出当前已加载日志。未知 GPU 占用显示「占用未知」，缺失数值显示 `—`。API 异常显示错误，已读取内容保留供参考。

代码结构：`api.ts` 管理请求和轮询，`data.tsx` 提供共享资源，`components.tsx` 提供公共组件，`pages.tsx` 为资源页面，`RunWizard.tsx` 和 `RunDetail.tsx` 为训练流程。
