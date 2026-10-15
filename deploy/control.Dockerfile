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
RUN pip install --no-cache-dir uv==0.12.5 && uv sync --locked --no-dev --no-editable \
    && useradd --system --uid 10001 --create-home vla
COPY --from=web /web/dist/ apps/web/dist/
USER vla
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["vla-api"]
