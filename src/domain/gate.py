"""Step 8, the gate: what is shown as found, and when nothing is, the closest row and why it is not.

A row a specialist confirmed passes; any other passes when no hard rule rejected it and its score is at the
threshold or above — and the choice model, if asked, did not turn it down. Of those the best top-K are
shown, the same position first. Only unverified rows passed: shown, but the answer is «не найдено».
"""

from __future__ import annotations

from enum import StrEnum

from domain.frozen import Frozen
from domain.knowledge import Category, Level
from domain.matching import BY_TEXT, Assessment, Candidate, Decision, Kind, Matcher, Outcome
from domain.rules import Gate


class Reason(StrEnum):
    UNKNOWN_CATEGORY = "unknown_category"
    CATEGORY_ABSENT = "category_absent"
    HARD_MISMATCH = "hard_mismatch"
    LOW_SCORE = "low_score"
    UNVERIFIED = "unverified"
    DISMISSED = "dismissed"


class NotFound(Frozen):
    reason: Reason
    message: str
    closest: Assessment | None


class Gatekeeper:
    def __init__(self, gate: Gate, matcher: Matcher):
        self._gate = gate
        self._matcher = matcher

    @staticmethod
    def confirmed(assessed: list[Assessment]) -> list[Assessment]:
        """What a specialist confirmed: it passes whatever the score."""
        return [a for a in assessed if a.confirmed in (Decision.SAME, Decision.ANALOG)]

    def open(self, assessed: list[Assessment]) -> list[Assessment]:
        """Not decided by a specialist, not rejected by a rule, above the threshold: the model's to judge."""
        return [a for a in assessed if a.confirmed is None and self._above(a)]

    def answer(
        self,
        category: Category | None,
        passing: list[Assessment],
        assessed: list[Assessment],
        in_category: list[Assessment],
        nearest: list[Candidate],
    ) -> tuple[list[Assessment], NotFound | None]:
        """The rows shown, and why the answer is «не найдено» if it is."""
        accepted = sorted(
            (a for a in passing if not a.dismissed),
            key=lambda a: (self._gate.order.index(a.kind), a.confirmed is None, -a.score),
        )[: self._gate.top_k]
        if any(a.kind is not Kind.UNVERIFIED for a in accepted):
            return accepted, None
        if accepted:
            return accepted, self._unverified(category, accepted[0])
        # what would have passed but a specialist or the model turned down
        dismissed = passing + [a for a in assessed if a.dismissed and self._above(a)]
        if dismissed:
            return [], self._dismissed(max(dismissed, key=lambda a: a.score))
        return [], self._explain(category, in_category, nearest)

    def _above(self, assessed: Assessment) -> bool:
        return not assessed.rejected and assessed.score >= self._gate.threshold

    @staticmethod
    def _unverified(category: Category | None, closest: Assessment) -> NotFound:
        unchecked = [a.name for a in closest.unreadable] + [
            c.attribute.name for c in closest.checks_with(Outcome.MISSING) if c.attribute.level is Level.HARD
        ]
        return NotFound(
            reason=Reason.UNVERIFIED,
            message=f"В категории «{category.name if category else ''}» есть похожие позиции, но жёсткие характеристики "
            f"не удалось сравнить ({', '.join(unchecked)}): совпадение не подтверждено, проверьте вручную.",
            closest=closest,
        )

    @staticmethod
    def _dismissed(closest: Assessment) -> NotFound:
        by = "специалист отклонил" if closest.confirmed is Decision.NOT_ANALOG else "ИИ-проверка сочла"
        return NotFound(
            reason=Reason.DISMISSED,
            message=f"Похожие по тексту позиции есть, и правила их не отсекли, но {by} каждую другим товаром.",
            closest=closest,
        )

    def _explain(self, category: Category | None, assessed: list[Assessment], nearest: list[Candidate]) -> NotFound:
        nearest_row = max(nearest, key=self._matcher.text, default=None)
        closest_by_text = self._matcher.assess(BY_TEXT, nearest_row) if nearest_row else None
        if category is None:
            return NotFound(
                reason=Reason.UNKNOWN_CATEGORY,
                message="Не удалось определить, что это за товар: ни одна категория справочника не узнала запрос, "
                "а по тексту ничего достаточно близкого нет.",
                closest=closest_by_text,
            )
        if not assessed:
            return NotFound(
                reason=Reason.CATEGORY_ABSENT,
                message=f"Запрос распознан как «{category.name}», но таких позиций в истории закупок нет.",
                closest=closest_by_text,
            )
        passing = [a for a in assessed if not a.rejected]
        if not passing:
            closest = max(assessed, key=lambda a: (len(a.checks_with(Outcome.MATCH)), a.score))
            failed = "; ".join(
                f"{c.attribute.name}: {c.candidate} вместо {c.query}"
                for c in closest.checks_with(Outcome.MISMATCH)
                if c.attribute.level is Level.HARD
            )
            return NotFound(
                reason=Reason.HARD_MISMATCH,
                message=f"В категории «{category.name}» {len(assessed)} поз., ни одна не проходит жёсткие правила замены. "
                f"Ближайшая не подходит: {failed}.",
                closest=closest,
            )
        closest = max(passing, key=lambda a: a.score)
        return NotFound(
            reason=Reason.LOW_SCORE,
            message=f"В категории «{category.name}» нет позиции с оценкой выше порога {self._gate.threshold}: "
            "характеристики запроса у кандидатов не указаны или текст слишком далёк.",
            closest=closest,
        )
