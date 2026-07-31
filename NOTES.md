# 运行与部署说明

## 配置原则

生产环境必须显式提供模型配置和 `ALLOWED_ORIGIN`。后者只接受单个规范化的 HTTP/HTTPS origin；公网服务应使用自己的 HTTPS 地址，而不是开发地址。TLS 应由平台或反向代理终止。

将 API Key 仅放在平台 Secret 或受保护的运行时环境变量中。它不能进入 Docker 镜像层、静态 JavaScript、demo 文件、日志、trace 或提交历史。`TRUSTED_PROXY_CIDRS` 默认留空；仅当反向代理的出口 CIDR 已核实时才填写。不要无条件信任 `X-Forwarded-For`。

## 持久数据

`demo_workspace_seed/` 是版本化的只读初始化素材。`workspace/` 和 `traces/` 是运行时状态，应该放在备份策略覆盖的持久卷中。推荐平台上将卷挂到 `/data`，再用 `WORKSPACE_ROOT=/data/workspace`、`TRACE_ROOT=/data/traces` 指向它们。

不要把 `/app` 整体挂载为数据卷：它会遮蔽镜像中的应用、静态页面与 seed。服务以 UID 10001 运行，部署后需确认卷目录对该 UID 可写。

## Reset 故障处理

工作区 reset 使用 `reset journal v3` 记录复制、备份和安装阶段。正常情况下，下一次启动会继续或回滚可验证的状态。若遗留的旧格式状态表现为“空 workspace + 非空 staging”，服务会停止并要求人工处理；先保留卷快照，再检查 journal、staging、backup 和 workspace 的内容，确认目标后再删除或恢复。

## 扩展前提

当前服务是单进程、单工作区、内存限流、非多租户实现。重启会清空限流数据；多个 worker 或多个副本没有共享的工作区锁、会话状态或限流器。若要做多用户、长期记忆或 RAG，应先补齐身份认证、租户隔离、外部持久队列/锁、数据库迁移、向量检索权限和审计策略。

工作区内容是非信任输入。六个工具仅能列目录、检索、读取、查看元数据、写入和移动文件；服务不暴露 shell、删除、网络或数据库能力。即便如此，模型写入与移动造成的业务影响仍须由操作者复核。

## 发布检查

发布前在 Python 3.12 环境执行 Python 测试、前端 Node 测试、编译检查和差异检查。若本机只有 Python 3.11，测试结果只能视为兼容性信号，不能宣称已完成 Python 3.12 验证。Docker 可用时，再构建镜像、启动容器、请求 `/health`，并确认进程 UID 为 10001。
