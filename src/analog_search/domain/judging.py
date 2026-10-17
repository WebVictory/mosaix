"""The choice model as a judge of what the rules let through but could not settle.

The rules decide first: a candidate with a hard mismatch (other cores, section, voltage) is rejected and
never shown to the model. Of the rest, the model judges only the unresolved ones: a row of the category
whose hard attributes could not all be compared, a row of another category found by text, any row of
a query with no category. Each is «та же», «аналог», «другой товар» or NONE (cannot tell from the names).
A sure answer is kept, within limits the rules set:
- «другой товар» dismisses the candidate;
- «аналог» or «та же» makes an unverified candidate an analog; «та же» is kept as the same only for a row
  of the query's category whose compared attributes are all identical: the model never overrides
  a difference the rules found, and it never confirms a category the rules did not;
- a row of a query with no category stays found by text, the verdict is only shown.
An unsure answer, or none (the model is down), leaves the candidate as the rules left it.
"""

from __future__ import annotations

from dataclasses import replace

from analog_search.domain.matching import Assessment, Kind, Outcome, Verdict
from analog_search.domain.reading import NONE, ChoiceReader, Question

UNRESOLVED = (Kind.UNVERIFIED, Kind.TEXT)
OPTIONS = {
    Verdict.SAME: "та же позиция: то же назначение и те же характеристики",
    Verdict.ANALOG: "аналог: другая марка или производитель, но подходит на замену",
    Verdict.OTHER: "не подходит: другой товар или другие характеристики",
    NONE: "по названиям нельзя понять",
}


class Judge:
    def __init__(self, reader: ChoiceReader, min_confidence: float, top: int = 10):
        self._reader = reader
        self._min_confidence = min_confidence
        self._top = top

    def judge(self, query: str, category_name: str | None, assessed: list[Assessment]) -> list[Assessment]:
        """assessed with the verdicts applied; only the best `top` unresolved candidates are asked about.
        Raises ReaderError."""
        asked = sorted((a for a in assessed if a.kind in UNRESOLVED), key=lambda a: -a.score)[: self._top]
        if not asked:
            return assessed
        questions = {
            f"k{i}": Question(
                f"k{i}",
                f"Позиция из истории закупок: «{a.candidate.name}». Подходит ли она на замену запрошенной?",
                {str(option): text for option, text in OPTIONS.items()},
                {},
            )
            for i, a in enumerate(asked)
        }
        state = f"Запрос покупателя: «{query}»" + (f". Вид продукции: {category_name}" if category_name else "")
        choices = self._reader.choose(state, list(questions.values()))
        verdicts = {}
        for code, assessment in zip(questions, asked, strict=True):
            choice = choices.get(code)
            if choice and choice.option in Verdict and choice.confidence >= self._min_confidence:
                verdicts[id(assessment)] = Verdict(choice.option)
        return [self._apply(a, verdicts[id(a)]) if id(a) in verdicts else a for a in assessed]

    @staticmethod
    def _apply(assessment: Assessment, verdict: Verdict) -> Assessment:
        if verdict is Verdict.OTHER or assessment.kind is Kind.TEXT:
            return replace(assessment, verdict=verdict)
        identical = (
            not assessment.foreign
            and not assessment.unreadable
            and bool(assessment.checks)
            and all(c.outcome is Outcome.MATCH and c.exact for c in assessment.checks)
        )
        kind = Kind.SAME if verdict is Verdict.SAME and identical else Kind.ANALOG
        return replace(assessment, verdict=verdict, kind=kind)
