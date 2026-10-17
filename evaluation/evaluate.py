"""Gold queries → our search and the baseline → evaluation/metrics_report.md.

uv run python evaluation/evaluate.py [--gold FILE] [--out FILE] [--llm off|cache|live|auto]

Runs on the workbook and research/*.yaml in memory: no database; no external keys unless --llm live.
The candidate store is the database store's twin with the same pg_trgm similarity.
With the choice model and the embeddings (cache: kept in data/llm, live: asked and kept there) the search
runs twice, with them and by the rules alone, so the report shows what they add.

hit@1, hit@5   — share of match queries with an expected row first / in the top 5
misplace@1     — share of queries with a key attribute whose first row is not an expected one:
                 expected_ids lists every row with the right key value, so it has another one
no_match_ok    — share of no_match queries answered «не найдено»
false_found    — share of no_match queries answered with candidates
By group: correct answers (an expected row first, or «не найдено» when nothing should be found).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from baseline import Baseline

from analog_search.app.bootstrap import choice_reading, models
from analog_search.app.config import Settings
from analog_search.domain.judging import Judge
from analog_search.infrastructure import scoring_yaml
from analog_search.infrastructure.knowledge import positions_xlsx, yaml_files
from analog_search.repositories.memory import MemoryStore
from analog_search.services.knowledge import embed, parse, read_hard
from analog_search.services.search import SearchService

HERE = Path(__file__).parent


@dataclass(frozen=True)
class Answer:
    rows: list[int]  # empty — «не найдено»
    top: str  # the first row's name, or why nothing was found


@dataclass(frozen=True)
class Gold:
    id: str
    group: str
    query: str
    expected_kind: str
    expected_ids: list[int]
    key: str | None = None
    note: str | None = None


def metrics(gold: list[Gold], answers: dict[str, Answer]) -> dict[str, float]:
    match = [g for g in gold if g.expected_kind == "match"]
    keyed = [g for g in match if g.key]
    no_match = [g for g in gold if g.expected_kind == "no_match"]

    def share(items: list[Gold], ok) -> float:
        return sum(ok(g, answers[g.id]) for g in items) / len(items) if items else 0.0

    return {
        "hit@1": share(match, lambda g, a: bool(a.rows) and a.rows[0] in g.expected_ids),
        "hit@5": share(match, lambda g, a: bool(set(a.rows[:5]) & set(g.expected_ids))),
        "misplace@1": share(keyed, lambda g, a: bool(a.rows) and a.rows[0] not in g.expected_ids),
        "no_match_ok": share(no_match, lambda g, a: not a.rows),
        "false_found": share(no_match, lambda g, a: bool(a.rows)),
    }


LOWER_IS_BETTER = {"misplace@1", "false_found"}


def report(gold: list[Gold], runs: dict[str, dict[str, Answer]], info: dict[str, str]) -> str:
    """runs — answers by column title: ours first, the baseline last; Δ is the first against the last."""
    titles = list(runs)
    computed = {t: metrics(gold, runs[t]) for t in titles}
    ours, base = computed[titles[0]], computed[titles[-1]]
    lines = [
        "# Метрики поиска аналогов",
        "",
        "Сгенерировано `evaluation/evaluate.py`; не править руками.",
        "",
        *(f"- {k}: {v}" for k, v in info.items()),
        "",
        f"| Метрика | {' | '.join(titles)} | Δ |",
        "|---|" + "---|" * (len(titles) + 1),
    ]
    for name in ours:
        delta = ours[name] - base[name]
        better = delta < 0 if name in LOWER_IS_BETTER else delta > 0
        values = " | ".join(f"{computed[t][name]:.2f}" for t in titles)
        lines.append(f"| {name} | {values} | {delta:+.2f}{' ✓' if better else ''} |")
    groups = list(dict.fromkeys(g.group for g in gold))
    lines += [
        "",
        "## По группам: верных ответов",
        "",
        f"| Группа | Запросов | {' | '.join(titles)} |",
        "|---|---|" + "---|" * len(titles),
    ]
    for group in groups:
        items = [g for g in gold if g.group == group]
        cells = " | ".join(f"{sum(_correct(g, runs[t][g.id]) for g in items)}" for t in titles)
        lines.append(f"| {group} | {len(items)} | {cells} |")
    lines += [
        "",
        "## По запросам",
        "",
        "| id | Запрос | Ожидание | " + " | ".join(f"{t}: top-1 | Верно" for t in titles) + " |",
        "|---|---|---|" + "---|---|" * len(titles),
    ]
    for g in gold:
        expected = "не найдено" if g.expected_kind == "no_match" else ", ".join(map(str, g.expected_ids))
        cells = " | ".join(f"{_top(runs[t][g.id])} | {_verdict(g, runs[t][g.id])}" for t in titles)
        lines.append(f"| {g.id} | {g.query} | {expected} | {cells} |")
    first = runs[titles[0]]
    failures = [g for g in gold if not _correct(g, first[g.id])]
    lines += ["", f"## Ошибки: {titles[0]}", ""]
    lines += [f"- **{g.id}** «{g.query}»: {_top(first[g.id])}. {g.note or ''}".rstrip() for g in failures] or ["нет"]
    return "\n".join(lines) + "\n"


def _top(answer: Answer) -> str:
    return f"{answer.rows[0]}: {answer.top[:60]}" if answer.rows else f"не найдено ({answer.top})"


def _correct(g: Gold, answer: Answer) -> bool:
    if g.expected_kind == "no_match":
        return not answer.rows
    return bool(answer.rows) and answer.rows[0] in g.expected_ids


def _verdict(g: Gold, answer: Answer) -> str:
    if _correct(g, answer):
        return "✓"
    if g.expected_kind == "match" and set(answer.rows[:5]) & set(g.expected_ids):
        return "в top-5"
    return "✗"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=HERE / "gold_queries.json")
    parser.add_argument("--out", type=Path, default=HERE / "metrics_report.md")
    parser.add_argument(
        "--llm", choices=["off", "cache", "live", "auto"], help="choice model mode (default: ANALOG_LLM)"
    )
    args = parser.parse_args()
    settings = Settings(**({"llm": args.llm} if args.llm else {}))

    gold = [Gold(**g) for g in json.loads(args.gold.read_text(encoding="utf-8"))]
    knowledge = yaml_files.read(settings.knowledge_dir)
    records = positions_xlsx.read(settings.positions_file)
    config = scoring_yaml.read(settings.search_config)
    positions = parse(knowledge, records)
    info = {
        "Запросов": f"{len(gold)} (match {sum(g.expected_kind == 'match' for g in gold)}, "
        f"no_match {sum(g.expected_kind == 'no_match' for g in gold)})",
        "Справочник": knowledge.version,
        "Порог": str(config.threshold),
    }

    runs: dict[str, dict[str, Answer]] = {}
    used = models(settings)
    reading = choice_reading(settings, knowledge, used)
    if used is not None and reading is not None:
        read, counts = read_hard(knowledge, positions, reading, settings.llm_workers)
        vectors = embed(positions, used.embedder)
        with_model = SearchService(
            knowledge,
            MemoryStore(knowledge.version, read, vectors),
            config,
            reading,
            settings.llm_mode,
            embedder=used.embedder if vectors else None,
            judge=Judge(used.reader, settings.llm_min_confidence, config.judge_k),
        )
        runs["Правила + ИИ"], elapsed_ms = _run(with_model, gold)
        info["ИИ"] = (
            f"{settings.llm_model}, режим {settings.llm_mode}, порог уверенности {settings.llm_min_confidence} "
            f"(число, найденное в тексте, — {settings.llm_min_confidence_stated}); "
            f"база: спрошено {counts['llm_asked']} поз., прочитано {counts['llm_read']}, ошибок {counts['llm_failed']}"
        )
        info["Эмбеддинги"] = (
            f"{settings.embedding_model}, векторов позиций {len(vectors)}"
            if vectors
            else "нет: поиск только по написанию"
        )
        info["Среднее время запроса с ИИ (в памяти)"] = f"{elapsed_ms:.0f} мс"
    rules = SearchService(knowledge, MemoryStore(knowledge.version, positions), config)
    runs["Только правила"], elapsed_ms = _run(rules, gold)
    info["Среднее время запроса по правилам (в памяти)"] = f"{elapsed_ms:.0f} мс"
    baseline, base = Baseline(records), {}
    for g in gold:
        hits = baseline.search(g.query)
        base[g.id] = Answer([h.row for h in hits], hits[0].name if hits else "")
    runs["Baseline (поиск по словам)"] = base

    text = report(gold, runs, info)
    args.out.write_text(text, encoding="utf-8")
    print(text)


def _run(service: SearchService, gold: list[Gold]) -> tuple[dict[str, Answer], float]:
    answers, started = {}, time.perf_counter()
    for g in gold:
        result = service.search(g.query)
        answers[g.id] = (
            Answer([a.candidate.row for a in result.candidates], result.candidates[0].candidate.name)
            if result.found
            else Answer([], result.not_found.reason if result.not_found else "")
        )
    return answers, (time.perf_counter() - started) * 1000 / len(gold)


if __name__ == "__main__":
    main()
