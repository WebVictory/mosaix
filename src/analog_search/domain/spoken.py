"""Numbers written in words, as people type them in a query: «семь жил по полтора», «два с половиной», «двухполюсный».

Mostly used to check a value the model chose: «три жилы по два с половиной» states 3 and 2.5. A lone
number in words is not extracted by rules: what «два» refers to is the model's reading, the check only
confirms that the number is in the text. One spelling is unambiguous and is written in digits: a size,
«сто на четыреста» → «100 на 400», read then like «100x400».
"""

from __future__ import annotations

import re

_UNITS = {
    1: "один одна одно одного одной одном одну",
    2: "два две двух двум двумя",
    3: "три трех трем тремя",
    4: "четыре четырех четырем четырьмя",
    5: "пять пяти пятью",
    6: "шесть шести шестью",
    7: "семь семи семью",
    8: "восемь восьми восемью",
    9: "девять девяти девятью",
    10: "десять десяти десятью",
    11: "одиннадцать одиннадцати",
    12: "двенадцать двенадцати",
    13: "тринадцать тринадцати",
    14: "четырнадцать четырнадцати",
    15: "пятнадцать пятнадцати",
    16: "шестнадцать шестнадцати",
    17: "семнадцать семнадцати",
    18: "восемнадцать восемнадцати",
    19: "девятнадцать девятнадцати",
}
_TENS = {
    20: "двадцать двадцати",
    30: "тридцать тридцати",
    40: "сорок сорока",
    50: "пятьдесят пятидесяти",
    60: "шестьдесят шестидесяти",
    70: "семьдесят семидесяти",
    80: "восемьдесят восьмидесяти",
    90: "девяносто девяноста",
}
_HUNDREDS = {
    100: "сто ста",
    200: "двести двухсот",
    300: "триста трехсот",
    400: "четыреста четырехсот",
    500: "пятьсот пятисот",
}
_HALF = {1.5: "полтора полторы полутора"}
# «двухполюсный», «трехжильный», «одножильный»: the count is the first part of the word
_PREFIXES = {1: "одно", 2: "двух", 3: "трех", 4: "четырех", 5: "пяти", 7: "семи"}


def _forms(table: dict) -> dict[str, float]:
    return {form: float(value) for value, forms in table.items() for form in forms.split()}


_UNIT_WORDS, _TEN_WORDS, _HUNDRED_WORDS, _HALF_WORDS = map(_forms, (_UNITS, _TENS, _HUNDREDS, _HALF))
_WORD = re.compile(r"[а-я]+")
_NUMBER_WORD = "(?:{})(?![а-я])".format(
    "|".join(sorted({**_UNIT_WORDS, **_TEN_WORDS, **_HUNDRED_WORDS, **_HALF_WORDS}, key=len, reverse=True))
)
_SIDE = rf"(?:\d+(?:[.,]\d+)?|{_NUMBER_WORD}(?:\s+{_NUMBER_WORD})*(?:\s+с\s+половиной)?)"
_SIZE = re.compile(rf"(?<![а-я\d])({_SIDE})(\s+(?:на|x|х)\s+)({_SIDE})(?![а-я])")


def spoken_sizes(text: str) -> str:
    """«сто на четыреста», «пять на 95» → «100 на 400», «5 на 95» in a normalized text; other words stay."""

    def digits(side: str) -> str | None:
        if side[0].isdigit():
            return side
        numbers = spoken_numbers(side)
        return f"{numbers[0]:g}" if len(numbers) == 1 else None

    def size(found: re.Match[str]) -> str:
        left, right = digits(found[1]), digits(found[3])
        return f"{left}{found[2]}{right}" if left and right else found[0]

    return _SIZE.sub(size, text)


def spoken_numbers(text: str) -> list[float]:
    """Numbers stated in words in a normalized text (lower case, ё → е)."""
    words = _WORD.findall(text)
    numbers: list[float] = []
    i = 0
    while i < len(words):
        value, i = _number(words, i)
        if value is None:
            numbers.extend(float(n) for n, prefix in _PREFIXES.items() if words[i].startswith(prefix))
            i += 1
            continue
        if words[i : i + 2] == ["с", "половиной"]:
            value, i = value + 0.5, i + 2
        numbers.append(value)
    return numbers


def _number(words: list[str], i: int) -> tuple[float | None, int]:
    """«сто двадцать пять» → 125: hundreds, tens, units in this order, each optional."""
    if words[i] in _HALF_WORDS:
        return _HALF_WORDS[words[i]], i + 1
    value, start = 0.0, i
    for table in (_HUNDRED_WORDS, _TEN_WORDS, _UNIT_WORDS):
        if i < len(words) and words[i] in table:
            value, i = value + table[words[i]], i + 1
    return (value, i) if i > start else (None, i)
