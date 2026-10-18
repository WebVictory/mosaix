FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY src ./src
COPY research ./research
COPY static ./static
COPY demo ./demo
COPY evaluation/gold_queries.json ./evaluation/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
# Without the key the model and the query's vector come from data/llm only (README, «Запуск»). Then the schema,
# the history parsed with the current knowledge (idempotent upsert), the demo verdicts (a reload changes
# nothing), the API
CMD ["sh", "-c", "[ -n \"$OPENROUTER_API_KEY\" ] || echo 'OPENROUTER_API_KEY is not set: rules and kept model answers only' >&2; analog-search db init && analog-search positions load && analog-search decisions load && analog-search serve --host 0.0.0.0 --port 8000"]
