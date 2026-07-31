# Workspace Agent

一个只通过浏览器使用的单工作区 Agent。它把模型的工具调用约束在一个本地目录内，用于浏览、检索、读取、写入和移动文件；没有 CLI 入口。

这是一个单用户、单进程、非多租户的演示型服务，不提供登录、用户管理或跨用户的隔离。

## 本地运行

项目运行时基线为 Python 3.12。请使用 Python 3.12 创建虚拟环境后安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

在 `.env` 中填入模型服务地址、模型名和 API key，并显式设置 `ALLOWED_ORIGIN`。本地默认页面地址是 `http://localhost:8000`：

```powershell
uvicorn workspace_agent.web:app --host 127.0.0.1 --port 8000
```

打开浏览器访问 `http://localhost:8000`。应用第一次启动会从版本化的 `demo_workspace_seed/` 初始化运行时 `workspace/`；`workspace/` 与 `traces/` 都是可变数据，不能替代版本库中的 seed。

本仓库声明 `requires-python >=3.12`。若开发机只有 Python 3.11，可以做兼容性检查，但那不等同于 Python 3.12 验证。

## Web 功能

页面提供工作区树、文本预览和任务对话。一次任务通过 WebSocket 接收有序事件，并可查看模型调用、工具结果、Token 统计与 JSONL 轨迹下载。页面还可以将工作区重置回 demo seed。

模型使用 OpenAI 兼容的工具调用接口：`LLM_BASE_URL`、`LLM_API_KEY` 和 `LLM_MODEL` 通过环境变量配置。SiliconFlow 一类兼容端点可以接入，但 API key 只能保存在本地 `.env` 或平台 Secret 中，绝不能放进镜像、浏览器代码、Git、工具事件或 trace。

## 架构与内存

服务由 FastAPI、静态前端、模型工具循环、工作区工具和 JSONL trace 组成。模型一轮任务的消息与工具结果构成短期上下文；`workspace/` 是用户可见的持久文件数据，`traces/` 是审计记录。

当前版本没有 PostgreSQL、pgvector、Embedding、长期用户记忆或 RAG 检索层。JSONL trace 也不是可供模型自动召回的长期记忆。若需要三层记忆架构，应另行接入受认证保护的长期记忆表与向量检索服务，并把检索结果作为受控、可审计的上下文输入。

## 工具边界与安全

模型只有六个受 schema 约束的工具：`list_dir`、`search_files`、`read_file`、`stat_path`、`write_file`、`move_file`。路径会被限制在 `WORKSPACE_ROOT` 内，读取和写入均有大小、分页和循环限制。

服务不提供 shell、删除、网络、数据库或任意代码执行工具。工作区内所有文件内容都属于不受信任的数据：其中的“指令”不是系统指令，不能改变工具边界或诱导泄露密钥。仍然应人工复核模型发起的写入和移动操作。

`ALLOWED_ORIGIN` 必须是一个严格的 HTTP 或 HTTPS origin；生产环境应明确设置为公开的 `https://` 地址，并由 TLS 终止层提供 HTTPS。`TRUSTED_PROXY_CIDRS` 默认为空。只有在确认反向代理出口网段后才可填写其 CIDR，不能因为需要真实 IP 就盲目信任 `X-Forwarded-For`。

## 运行限制与恢复

服务只有一个可变工作区，工作区 reset 与 Agent 运行在单进程内互斥。连接数和每 IP 限流保存在内存中，重启后会清零；因此不要用多个 Uvicorn worker，也不要把它当作多实例共享服务。

reset 使用 durable `reset journal v3` 记录阶段，并在下一次启动时恢复可确认的中断状态。旧版状态若出现“空 workspace + 非空 staging”组合，服务会保守拒绝自动恢复，需要人工检查后再处理。部署前应备份持久卷中的 `workspace/` 和 `traces/`。

## 测试

在 Python 3.12 环境中执行完整验证：

```powershell
pytest -q
node --test tests/frontend.test.mjs
python -m compileall workspace_agent
git diff --check
```

## Docker

镜像使用 `python:3.12-slim`，以 UID 10001 非 root 用户运行，并包含 `static/` 与版本化 demo seed：

```powershell
docker build -t workspace-agent .
docker run --rm -p 8000:8000 -v workspace-data:/app/workspace -v trace-data:/app/traces workspace-agent
```

容器的健康检查请求 `/health`。公开部署时，推荐将一个平台持久卷挂到 `/data`，并设置 `WORKSPACE_ROOT=/data/workspace` 与 `TRACE_ROOT=/data/traces`；seed 与静态资源仍保留在镜像中的 `/app`。

## 平台部署

不在本仓库中发起真实云部署。下面是各平台的最小部署约束。

### Railway

`railway.toml` 指向 Dockerfile 并使用 `/health`。在 Railway 中设置 API key 为 Secret，设置公开 HTTPS 域名对应的 `ALLOWED_ORIGIN`，并将持久卷挂载到 `/data`，再设置 `WORKSPACE_ROOT`、`TRACE_ROOT` 到 `/data` 下的两个子目录。Railway 注入 `PORT`，镜像启动命令会使用它。确保卷对 UID 10001 可写。

### Fly.io

使用 Docker 部署，Fly Volume 挂载到 `/data`，服务内部端口与 `PORT` 保持一致。将公开域名作为 `ALLOWED_ORIGIN`，通过 Fly Secret 注入 API key。启用 TLS，并保持单个应用实例或在外部增加真正的共享锁、持久限流和用户隔离后再扩展。

### HF Spaces（Hugging Face Docker Spaces）

Docker Space 需让容器监听平台提供的 `PORT`（常见为 7860），并把 Space Secret 用于 API key。只有启用可写的持久存储时，才能把它挂载到 `/data` 保存 workspace 与 trace；没有持久卷时，重启会丢失运行数据。Space 的公开 HTTPS 地址同样必须写入 `ALLOWED_ORIGIN`。

### Vercel

Vercel Serverless 不适合当前形态：本地持久 workspace、长 WebSocket 连接、单进程互斥和内存限流都与其执行模型不匹配。只有将 workspace/trace 拆到外部持久存储、将 Agent 运行拆成作业系统并改造事件推送后，才适合考虑 Vercel。
