"""What a query being typed may become: its last word finished, the names it is the start of, the categories it names."""

from __future__ import annotations

from analog_search.domain.completion import Completer
from analog_search.domain.knowledge import Category

NAMES = [
    "Кабель контрольный КВВГнг 7х1,5",
    "Кабель контрольный КВВГнг 4х2,5",
    "Кабель силовой ВВГнг 3х2,5 с контактами",
    "Оповещатель световой",
    "Светильник светодиодный",
    "Светильник переносной",
    "Ворота откатные",
    "Модуль отказоустойчивости",
]
CABLE = Category("kabel_prochiy", "кабель (прочий)", "кабели и провода", triggers=("кабель",))
CONTROL = Category("kabel_kontrolny", "кабель контрольный", "кабели и провода", triggers=("кабель контрольный", "кввг"))
LIGHT = Category("svetilnik", "светильник", "освещение", triggers=("светильник",))
COMPLETER = Completer(NAMES, [CABLE, CONTROL, LIGHT])


def test_the_last_word_is_finished_with_what_follows_the_previous_one() -> None:
    assert COMPLETER.words("кабель конт", 3) == ["кабель контрольный"]  # «контактами» never follows «кабель»
    assert COMPLETER.words("Ворота отк", 3) == ["Ворота откатные"]


def test_without_a_previous_word_the_most_frequent_comes_first() -> None:
    assert COMPLETER.words("Свет", 2) == ["Светильник", "Световой"]
    assert COMPLETER.words("с", 3) == []  # too short to finish


def test_a_name_starting_with_the_query_comes_before_one_holding_it() -> None:
    names = COMPLETER.names("свет", 3)

    assert names[:2] == ["Светильник переносной", "Светильник светодиодный"]
    assert "Оповещатель световой" in names


def test_categories_named_by_the_query_or_started_by_it() -> None:
    assert COMPLETER.categories("кабель конт", 3) == [CONTROL, CABLE]  # the one being typed first
    assert COMPLETER.categories("кабель контрольный 7х1,5", 3) == [CONTROL, CABLE]  # the longer label named first
    assert COMPLETER.categories("квВГ 7х1,5", 3) == [CONTROL]  # a trigger mark, in any case
    assert COMPLETER.categories("свети", 3) == [LIGHT]
    assert COMPLETER.categories("ворота", 3) == []
