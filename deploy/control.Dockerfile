FROM node:22-alpine AS web
WORKDIR /web
RUN corepack enable
COPY apps/web/package.json apps/web/pnpm-lock.yaml apps/web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY apps/web/ ./
RUN pnpm build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 VLA_HOST=0.0.0.0 VLA_PORT=8000
COPY pyproject.toml uv.lock README.md ./
COPY apps/control_plane/ apps/control_plane/
COPY apps/agent/ apps/agent/
COPY packages/ packages/
COPY scripts/build_agent_bundle.py scripts/build_agent_bundle.py
COPY deploy/install-agent.sh deploy/install_agent.py deploy/agent_ownership.py deploy/
RUN pip install --no-cache-dir uv==0.12.5 && uv sync --locked --no-dev --no-editable \
    && .venv/bin/python scripts/build_agent_bundle.py \
    && useradd --system --uid 10001 --create-home vla \
    && mkdir -p /var/lib/vla && chown vla:vla /var/lib/vla
COPY --from=web /web/dist/ apps/web/dist/
USER vla
ENV PATH="/app/.venv/bin:$PATH" VLA_AGENT_BUNDLE=/app/.local/agent-dist/vla-agent-bundle.tar.gz
EXPOSE 8000
CMD ["vla-api"]
