import json, os, re
from collections import Counter, defaultdict
import cats, attrspec

OUT = os.environ.get("RESEARCH_OUT", os.path.dirname(os.path.abspath(__file__)) + "/../")
cl = json.load(open("classified.json"))
rows = json.load(open("rows.json"))
cnt = Counter(r["cat"] for r in cl)
by = defaultdict(list)
for r in cl:
    by[r["cat"]].append(r)

def q(s):
    return json.dumps(str(s), ensure_ascii=False)

def scal(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, (int, float)):
        return str(v)
    return q(v)

def flow(lst):
    return "[" + ", ".join(scal(x) for x in lst) + "]"

def norm(s):
    return s.lower().replace("ё", "е")

total = len(cl)
service = sum(v for k, v in cnt.items() if k.startswith("служебная"))
other = cnt.get("прочее", 0)
work = total - service
covered = work - other

# ---------- categories.yaml ----------
cat_names = {c[0] for c in cats.CATS}
L = []
L.append("версия: 1")
L.append("база: " + q("analog-search/data/positions.xlsx, лист «База позиций»"))
L.append("итого:")
L.append(f"  строк_всего: {total}")
L.append(f"  служебных_строк: {service}")
L.append(f"  товарных_строк: {work}")
L.append(f"  не_классифицировано: {other}")
L.append(f"  покрытие_товарных_процент: {round(100*covered/work,1)}")
L.append("правила_сопоставления: " + q("триггер ищется с начала слова в нормализованном тексте (нижний регистр, ё -> е); побеждает самое раннее вхождение, при равенстве - самый длинный триггер; анти-триггер отменяет совпадение, если перекрывает его позицию"))
L.append("категории:")

def emit_cat(name, group, trig, anti, syn, in_base):
    n = cnt.get(name, 0)
    L.append(f"  - имя: {q(name)}")
    L.append(f"    группа: {q(group)}")
    L.append(f"    в_базе: {'да' if n > 0 else 'нет'}")
    L.append(f"    строк_в_базе: {n}")
    L.append(f"    триггеры: {flow(sorted(dict.fromkeys(t.strip() for t in trig), key=lambda t: -len(t)))}")
    L.append(f"    анти_триггеры: {flow(anti)}")
    L.append(f"    синонимы: {flow(syn)}")
    if n:
        ex = by[name][0]
        L.append("    пример:")
        L.append(f"      строка: {ex['row']}")
        L.append(f"      текст: {q(ex['name'])}")
    else:
        L.append("    пример: null")

for c in cats.CATS:
    emit_cat(c[0], c[1], c[2], c[3], c[4], c[5])
outside = [o for o in attrspec.OUTSIDE if o[0] not in cat_names]
for o in outside:
    emit_cat(o[0], o[1], o[2], o[3], o[4], False)
open(OUT + "categories.yaml", "w").write("\n".join(L) + "\n")

# ---------- attributes.yaml ----------
L = []
L.append("версия: 1")
L.append("статус_по_умолчанию: предложено")
L.append("шкала_уровней:")
L.append("  hard: " + q("несовпадение исключает кандидата из аналогов (может быть показан как 'ближайший, не аналог')"))
L.append("  soft: " + q("несовпадение снижает оценку, кандидат остается"))
L.append("  info: " + q("показывается человеку, в оценке не участвует"))
L.append("шкала_сравнения:")
for k, v in [("равно", "значения должны совпасть после нормализации"),
             ("не_хуже", "кандидат >= запроса по упорядоченному ряду или числу"),
             ("не_больше", "кандидат <= запроса (например SDR)"),
             ("ступень", "отличие не более допуск_ступеней шагов по ряду"),
             ("включает", "свойство кандидата покрывает запрошенное (диапазон, набор функций, исполнение по порядку)"),
             ("без_порядка", "многомерный размер сравнивается как отсортированный набор чисел")]:
    L.append(f"  {k}: {q(v)}")
L.append("шкала_уверенности: [\"источник\", \"здравый смысл\", \"догадка\"]")
L.append("категории:")

def emit_attrs(name, attrs, in_base_rows):
    L.append(f"  - категория: {q(name)}")
    L.append(f"    строк_категории_в_базе: {len(in_base_rows)}")
    L.append("    характеристики:")
    for a in attrs:
        L.append(f"      - характеристика: {q(a['key'])}")
        L.append(f"        величина: {q(a['величина'])}")
        L.append(f"        единица: {scal(a['единица'])}")
        L.append(f"        написания: {flow(a['написания'])}")
        L.append(f"        ряд: {flow(a['ряд']) if a['ряд'] is not None else 'null'}")
        L.append(f"        уровень: {a['уровень']}")
        L.append("        статус: предложено")
        L.append(f"        сравнение: {a['сравнение']}")
        if a["сравнение"] == "ступень":
            L.append(f"        допуск_ступеней: {a.get('допуск', 1)}")
        if "порядок" in a:
            L.append(f"        порядок: {flow(a['порядок'])}")
        L.append(f"        замена: {q(a['замена']) if a['замена'] else 'null'}")
        L.append(f"        источник: {q(a['источник']) if a['источник'] else q('нет')}")
        L.append(f"        уверенность: {q(a['уверенность'])}")
        if not in_base_rows:
            L.append(f"        строк_в_базе: {q('в наименованиях не встречается')}")
        elif a["rx"] is None:
            L.append(f"        строк_в_базе: {q('не подсчитано автоматически')}")
        else:
            k = sum(1 for r in in_base_rows if re.search(a["rx"], norm(r["name"])))
            L.append(f"        строк_в_базе: {k if k else q('в наименованиях не встречается')}")

for c in cats.CATS:
    name = c[0]
    if name in attrspec.SPEC:
        emit_attrs(name, attrspec.SPEC[name], by.get(name, []))
for o in attrspec.OUTSIDE:
    if o[0] in cat_names:
        emit_attrs(o[0], o[5], by.get(o[0], []))
    else:
        emit_attrs(o[0], o[5], [])
L.append("без_характеристик:")
L.append("  примечание: " + q("для этих категорий сравнение по характеристикам из наименования не предложено: товар определяется моделью или это не товар"))
L.append("  категории:")
for c in cats.CATS:
    if c[0] not in attrspec.SPEC and not any(o[0] == c[0] for o in attrspec.OUTSIDE):
        L.append(f"    - {q(c[0])}")
open(OUT + "attributes.yaml", "w").write("\n".join(L) + "\n")

# ---------- noise.yaml ----------
NOISE = [
 ("или_аналог", r"или аналог|или эквивалент|по типу|\(или анало", "оговорка о допустимости аналога; удалить до разбора, но можно сохранить флаг 'аналог допускается'", "удалить"),
 ("примечание_в_скобках", r"\(прим\.?", "примечание с альтернативными параметрами; не смешивать с основными характеристиками, выносить в отдельное поле", "вынести"),
 ("в_комплекте", r"в комплекте|комплектно с|в составе:?|в составе", "перечень состава; основное изделие стоит до этих слов", "отсечь хвост для поиска, сохранить для отображения"),
 ("ссылка_на_гост_ту", r"гост\s*[рp]?\s*[\d.-]+|\bту\s*[\d.-]+", "номер стандарта; не характеристика, но может уточнять тип изделия", "удалить из текста для сравнения"),
 ("имя_объекта", r"объект-[а-я]|гэс-\d|пс\s*\d+\s*кв|квл\s*\d+|вл\s*\d+\s*кв", "имя объекта и линии (КВЛ 220 кВ ГЭС-1 – Объект-А); числа здесь не характеристики изделия", "удалить"),
 ("единица_учета", r"\(\s*\d+\s*(?:шт|м|компл)\.?\s*\)|\bшт\.?\b|\bкомпл\.?\b", "количество и единица учета внутри наименования", "удалить"),
 ("служебные_слова", r"\bтипа\b|\bмарки\b|\bмарка\b|\bисполнения\b|\bизделие\b|\bизделия\b", "служебные слова-связки", "удалить"),
 ("тип_работ", r"монтаж|установка|прокладка|устройство", "слова работ; в базе товаров встречаются в смешанных строках", "проверять: если строка начинается с работы - это не товар"),
 ("масса_вес", r"\(\s*[\d.,]+\s*(?:кг|т)\s*\)|масса\s*[\d.,]+", "масса в скобках - справочно, не для сравнения", "вынести в info"),
 ("расход_на_площадь", r"[\d.,]+\s*кг/м2", "норма расхода материала из сметы, не свойство товара", "вынести в info"),
 ("кавычки_названия", r"[«»\"]", "торговое название в кавычках; оставить как слово, кавычки убрать", "нормализовать"),
 ("латиница_вместо_кириллицы", r"[a-z](?=[а-я])|(?<=[а-я])[a-z]|\bc[а-я]", "смешение латиницы и кириллицы в одном слове (Cвыше, КПCВВ, нг(A))", "заменить похожие латинские буквы на кириллицу в русских словах и наоборот в кодах"),
 ("опечатки", r"бронорован|кабел\b|огнестойкий кабел", "опечатки в наименовании", "словарь исправлений"),
 ("доп_цвет", r"\bбел(?:ый|ая|ое)\b|\bсер(?:ый|ая)\b|\bчерн(?:ый|ая)\b|ral\s*\d{4}", "цвет - обычно info", "вынести в info"),
]
L = ["версия: 1", "примечание: " + q("шаблоны - регулярные выражения по нормализованному тексту (нижний регистр, ё -> е); строк_в_базе посчитано по всей базе"), "шумы:"]
for key, rx, desc, act in NOISE:
    m = [r for r in rows if re.search(rx, norm(r["name"]))]
    L.append(f"  - имя: {q(key)}")
    L.append(f"    шаблон: {q(rx)}")
    L.append(f"    описание: {q(desc)}")
    L.append(f"    действие: {q(act)}")
    L.append(f"    строк_в_базе: {len(m)}")
    if m:
        L.append("    пример:")
        L.append(f"      строка: {m[0]['row']}")
        L.append(f"      текст: {q(m[0]['name'])}")
    else:
        L.append("    пример: null")
L.append("служебные_строки:")
L.append("  примечание: " + q("строки, которые не являются товаром; исключаются из поиска аналогов целиком"))
L.append("  категории:")
for c in cats.CATS:
    if c[0].startswith("служебная"):
        ex = by[c[0]][0]
        L.append(f"    - имя: {q(c[0])}")
        L.append(f"      триггеры: {flow(c[2])}")
        L.append(f"      строк_в_базе: {cnt[c[0]]}")
        L.append(f"      пример: {{строка: {ex['row']}, текст: {q(ex['name'])}}}")
open(OUT + "noise.yaml", "w").write("\n".join(L) + "\n")
print("total", total, "service", service, "work", work, "other", other, "cov", round(100*covered/work, 1), "outside", len(outside))
