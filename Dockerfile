FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV WORKSPACE_ROOT=/data/workspace \
    TRACE_ROOT=/data/traces

WORKDIR /app

COPY pyproject.toml README.md .env.example ./
COPY workspace_agent ./workspace_agent
COPY static ./static
COPY demo_workspace_seed ./demo_workspace_seed

RUN pip install --no-cache-dir . \
    && groupadd --gid 10001 workspace-agent \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin workspace-agent \
    && mkdir -p /data \
    && chmod 1777 /data

USER root

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD ["python", "-c", "import os; import urllib.request; from workspace_agent.container_entrypoint import resolve_port; assert urllib.request.urlopen('http://127.0.0.1:%d/health' % resolve_port(os.environ.get('PORT')), timeout=2).status == 200"]

ENTRYPOINT ["python", "-m", "workspace_agent.container_entrypoint"]
