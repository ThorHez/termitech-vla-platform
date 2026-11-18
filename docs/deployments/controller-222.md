# 222 中台部署与运维

2026-11-18，中台已从 `115.190.212.169` 迁移至 `root@58.251.254.222:/data1`。当前网页与 API 入口为 **[https://58.251.254.222:8443](https://58.251.254.222:8443)**。目标机的公网 80、443、8080 无法从外部访问，本次按用户选择使用 8443 和 Caddy 私有 CA 证书。

169 上的网页、控制服务已在切换时停止并禁用开机自启；169 仍作为训练节点运行，Agent、训练环境、数据集、模型及输出保留在原机器。此次没有关闭整台 169，也没有重启各节点的 Agent；验收时各 Agent PID 与切换前一致。

## HTTPS 访问

客户端需先核对并信任本次 Caddy 私有根证书。这是私有 CA 的公开根证书，不是公网受信 CA 签发的证书；仅使用 HTTPS 地址仍不足以让未配置的客户端自动信任它。

- 本次操作机上的证书：`.local/host-migration-20261118/Termitech-VLA-222-Root-CA.cer`。
- 根证书 SHA-256 指纹：`F7:2C:DF:54:FB:31:CC:30:E9:AA:18:CF:6F:08:79:F1:B2:44:A1:82:AD:6B:D8:70:15:A5:8C:34:82:F8:A6:3F`。
- 将核对后的根证书导入客户端信任库，再使用完整的 `https://58.251.254.222:8443` 地址登录。向其他客户端分发公开根证书即可，勿分发 CA 私钥或服务私钥。

## 运行布局

以下路径均位于 222，除非特别注明。

| 项目 | 实际位置或配置 |
| --- | --- |
| 应用目录 | `/data1/termitech-vla-platform` |
| SQLite 数据库 | `/data1/termitech-vla-platform/.local/control.db` |
| SSH 管理状态 | `/data1/termitech-vla-platform/.local/remote-management` |
| 控制服务 | `termitech-vla-control.service`，已启用并运行 |
| HTTPS 服务 | `termitech-vla-https.service`，已启用并运行 |
| 服务环境文件 | `/data1/termitech-vla-platform-migration-20261118/control.env` |
| 固定 Python 运行环境 | `/data1/termitech-vla-platform/.local/stage-estimator-validation-20261117.L67v64/control-venv/bin/python` |
| Caddy 配置 | `/etc/termitech-vla/https/Caddyfile` |
| Caddy 持久状态 | `/var/lib/termitech-vla-caddy` |
| 目标机兼容链接 | `/data4/termitech-vla-platform` → `/data1/termitech-vla-platform` |

控制服务继续使用迁移后的独立 Python 环境，不能改用目标机的系统 Python 3.10。兼容链接用于保留运行环境内的旧绝对路径，不是第二套部署。后续修改环境变量应核对 systemd 实际加载的上述 `control.env`，不要误改应用目录内的历史配置副本。

169 的 Agent 仍请求自己的 `http://127.0.0.1:8000`；该端口现在由 222 控制服务管理的反向 SSH 隧道提供，再转发至 222 的内部 API。其他节点保留原有隧道端口。169 的旧控制服务须保持停止，以免争用 8000 或出现两个中台同时管理节点。

## 日常检查与备份

在 222 上检查服务、内部健康接口及日志：

```sh
systemctl is-active termitech-vla-control.service termitech-vla-https.service
systemctl is-enabled termitech-vla-control.service termitech-vla-https.service
curl --fail http://127.0.0.1:8000/healthz
journalctl -u termitech-vla-control.service -u termitech-vla-https.service -n 100 --no-pager
```

从已信任根证书的外部客户端检查网页及 `/healthz`，并在平台中确认原节点 ID、心跳和训练记录。内部健康检查成功不代表公网 8443 或客户端证书信任已经正常。

数据库与整个 `remote-management` 必须一起备份；其中 `credentials.key` 用于解密数据库中的 SSH 凭据。还需保留当前环境文件、systemd 单元及 drop-in、Caddy 配置和持久状态。对运行中的 SQLite 使用 Backup API，或先停控制服务再生成一致性快照，不能只复制热数据库文件而遗漏 WAL。备份与环境文件按私密资料保存，不在日志或文档中输出凭据。

## 本次迁移验收基线

最终快照包含 37 张表，迁移验证未发现原记录 ID 缺失。以下是切换快照中的数量，服务上线后会继续增长：

| 记录 | 数量 |
| --- | ---: |
| 用户 | 7 |
| 训练任务 | 52 |
| 资产 | 265 |
| 训练事件 | 478,977 |
| 审计事件 | 5,626 |

应用及运行资料共复制 315,563 个文件系统条目，约 80 GB；zstd 压缩传输约 9 GB。训练节点上的实际数据集、模型和训练输出保持原位置，控制库中的来源路径与历史关联保留。

222 上的切换快照和验收资料位于 `/data1/termitech-vla-platform-migration-20261118/`，包括 `control-final.db`、`control-final.db.zst`、`final-manifest.json` 和 `acceptance.json`。169 保留 `/data4/termitech-vla-platform/.local/host-migration-20261118/` 中的源端迁移资料。切换快照用于追溯与核验，不代表上线后的最新状态。

## 回滚

222 已接收节点心跳和事件后，禁止直接恢复切换前快照。回滚需要：

1. 停止 222 的 HTTPS 和控制服务，确认其管理的隧道退出，保持 Agent 运行。
2. 备份 222 最新数据库及完整 `remote-management`，校验后私密同步回 169；保留 222 原件。
3. 将 169 节点当前安装记录的 `connection_mode`、`tunnel_state`、`tunnel_remote_port` 恢复为迁移前的 `direct`、`NOT_REQUIRED`、空端口，使 169 Agent 再次直接访问同机 API。其他节点的原隧道配置保持原值。
4. 核对 169 的原服务路径、环境配置及适用于其入口的有效证书，再启用控制与 HTTPS 服务，验证全部节点心跳、训练及事件连续性。

全过程仅允许一个控制服务管理这批节点；不要通过启动生产数据库副本进行演练，因为启动应用本身就会恢复 SSH 隧道并更新任务状态。
