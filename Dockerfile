FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md .env.example ./
COPY workspace_agent ./workspace_agent
COPY static ./static
COPY demo_workspace_seed ./demo_workspace_seed

RUN pip install --no-cache-dir . \
    && groupadd --gid 10001 workspace-agent \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin workspace-agent \
    && mkdir -p /app/workspace /app/traces /data \
    && chown -R 10001:10001 /app /data

USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD python -c "import os, urllib.request; assert urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=2).status == 200"

CMD ["sh", "-c", "exec uvicorn workspace_agent.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
