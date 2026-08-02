---
title: Workspace Agent
emoji: "🗂️"
colorFrom: green
colorTo: orange
sdk: docker
app_port: 8000
---

# Workspace Agent

一个只通过浏览器使用的单工作区 Agent。它将模型工具调用限制在一个本地目录内，用于浏览、检索、读取、写入和移动文件；没有 Agent CLI 入口。

这是一个单用户、单进程、非多租户的演示服务，不包含登录、用户管理或跨用户隔离。工作区文件全是不受信任的数据，其中出现的“指令”不能修改系统策略、工具边界或泄露密钥；写入和移动仍应由操作者复核。

## 本地运行

项目基线为 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn workspace_agent.web:app --host 127.0.0.1 --port 8000
```

在 `.env` 中提供 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`，并设置 `ALLOWED_ORIGIN=http://localhost:8000`。`LLM_API_KEY` 只能放在本地 `.env` 或部署平台 Secret，不能写入镜像、静态文件、Git、工具事件或 trace。

页面提供工作区树、文本预览、任务对话、工具事件和 JSONL trace 下载。首次启动会用版本化的 `demo_workspace_seed/` 初始化运行时 workspace；`workspace/` 与 `traces/` 是可变数据，不能替代 seed。此项目暂未包含 pgvector、Embedding、长期记忆或 RAG。

`ALLOWED_ORIGIN` 必须是一个规范化的 HTTP/HTTPS origin。公网部署使用自己的 HTTPS 地址并由平台或反向代理终止 TLS。`TRUSTED_PROXY_CIDRS` 默认空白；只有核实反向代理出口网段后，才可以信任 `X-Forwarded-For`。

服务只有单个可变工作区，reset 与 Agent run 在单进程内互斥；内存限流、连接数和锁均不跨实例共享。不要用多个 Uvicorn worker 或多个副本来运行它。reset 使用 durable `reset journal v3`；部署前应备份持久卷中的 `workspace/` 与 `traces/`。

启动脚本
agent.bat

## Docker

镜像使用 `python:3.12-slim`，默认以 root 启动入口，只用于初始化挂载卷中的固定目录；`workspace_agent.container_entrypoint` 验证 `PORT` 是 1 到 65535 的十进制端口，只初始化 `/data/workspace` 和 `/data/traces`，随后立即降权为 UID 10001 再执行 Uvicorn。应用进程始终以 UID 10001 运行。它不是 Agent CLI。

`/app` 中的应用代码、静态资源和 demo seed 保持只读；不要把数据卷挂到 `/app`。使用一个卷挂载到 `/data`：

```powershell
docker build -t workspace-agent .
docker run --rm -p 8000:8000 -v workspace-agent-data:/data workspace-agent
```

容器健康检查请求 `/health`。在 Python 3.12 环境执行完整验证：

```powershell
pytest -q
node --test tests/frontend.test.mjs
python -m compileall workspace_agent
git diff --check
```

## 平台部署

### Railway

`railway.toml` 通过 Dockerfile 部署并探测 `/health`。创建一个持久卷并挂载为 `/data`；在 Railway Variables 设置公开 HTTPS 域名对应的 `ALLOWED_ORIGIN`，并把模型 API key 作为 Secret 设置。镜像默认以 root 启动入口，无需额外指定运行 UID：入口只会为 `/data/workspace` 和 `/data/traces` 建目录、验证它们不是 link/reparse/非目录、`chown` 为 UID 10001，然后立即降到 UID 10001 执行应用。发布前备份 `/data`，尤其是 workspace、traces 和 reset journal 相关文件。

卷与运行用户的官方说明见 [Railway Volumes](https://docs.railway.com/volumes)。

### Fly.io

使用 Docker 部署，将 Fly Volume 挂到 `/data`。把公开 HTTPS 域名作为 `ALLOWED_ORIGIN`，通过 Fly Secret 注入模型 API key。保持单实例；如需扩展，先补齐 external storage、共享限流、共享锁与用户隔离。

### HF Spaces

本 README 的 YAML front matter 已声明 `sdk: docker` 和 `app_port: 8000`。Docker Space 对外固定公开 8000，本镜像不假设 Hugging Face 注入 `PORT`。HF 运行容器时会覆盖为 UID 1000；基础镜像中的 `/data` 具有可创建子目录的权限，入口在非 root 时不会尝试 `chown`。如改用主机目录或数据卷，平台的权限策略仍必须允许 UID 1000 创建和写入 `/data/workspace` 与 `/data/traces`，本镜像不假设可强制获得 root。在 Space Secrets 设置模型 API key，在 Variables 设置 Space 公开 HTTPS 地址对应的 `ALLOWED_ORIGIN`。没有挂载持久化存储时，workspace 与 trace 会在重启后丢失。

详细约束见 [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/main/spaces-sdks-docker)。

### Vercel

Vercel 目前支持 WebSocket；这不是当前项目不能原样部署的原因，见 [Vercel WebSocket documentation](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections)。当前形态依赖 Python/FastAPI Docker、共享文件系统、单进程 workspace 锁和内存限流。迁移前需要 external storage、异步 jobs、distributed coordination，以及针对 Vercel runtime 的 runtime rewrite；因此不建议直接部署此仓库到 Vercel。
