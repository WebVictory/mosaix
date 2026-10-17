# analog-search

Поиск аналогов материалов и оборудования по истории закупок: та же позиция, допустимый аналог с перечнем
отличий — или честное «не найдено» с причиной.

## Запуск

```bash
OPENROUTER_API_KEY=sk-or-... docker compose up --build
```

Открыть http://localhost:8000 (поиск), http://localhost:8000/knowledge (правка знаний), http://localhost:8000/docs (API).

Ключ OpenRouter обязателен: через него идут модель Jev Choice и вектор запроса (bge-m3); без ключа API не
стартует. Вместо параметра можно `cp .env.example .env` и вписать ключ туда. Первый старт восстанавливает базу из
дампа `data/dump/analog.dump`. Начать с чистой базы — `docker compose down -v`. Другой файл истории —
`POSITIONS_FILE=/path/file.xlsx`, другой порт — `API_PORT=8080`.

## Локально

```bash
uv sync
docker compose up -d db                               # PostgreSQL 17 + pg_trgm + pgvector, порт 5434
uv run analog-search db init && uv run analog-search positions load
uv run analog-search serve                            # http://127.0.0.1:8000
uv run analog-search --help                           # остальные команды: search, decisions, knowledge, sources

uv run pytest                                         # тесты без БД; -m integration — с БД
uv run ruff check . && uv run ruff format --check . && uv run mypy   # линтеры и типы
uv run python evaluation/evaluate.py --llm cache      # метрики → evaluation/metrics_report.md (без ключа, ответы ИИ из кэша)
```

## Где что лежит

| Путь | Что |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | схема и шаги поиска, где ИИ / код / человек, хранилище и модели, стоимость, качество, ограничения и план |
| [AI_LOG.md](AI_LOG.md) | журнал работы с ИИ и время |
| `src/analog_search/` | код по слоям: `domain` (правила и порты) ← `services` (сценарии) ← `repositories` (БД) и `infrastructure` (файлы, модели, источники) ← `app` (API, CLI, сборка); направление импортов проверяет `tests/test_layers.py` |
| `static/` | страницы поиска и правки знаний |
| `research/` | знания: `categories.yaml`, `attributes.yaml` (начальное наполнение БД), отчёт по базе, вопросы специалисту; `scripts/` — генератор yaml из базы |
| `config/search.yaml` | веса оценки, порог «не найдено», top-K |
| `evaluation/` | контрольный набор `gold_queries.json`, `evaluate.py`, отчёт |
| `demo/decisions.json` | демо-решения специалиста → золотые карточки |
| `data/` | `positions.xlsx` (копия базы из задания), дамп БД, кэш ответов модели |
| `prompts/` | промпт исследования предметной области |
| `tests/` | тесты |
