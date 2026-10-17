# Генерация research/*.yaml

`categories.yaml`, `attributes.yaml` и `noise.yaml` собираются из базы этими скриптами; `units.yaml` и `*.md` написаны вручную.

```bash
cd analog-search/research/scripts
python3 export_rows.py   # data/positions.xlsx -> rows.json
python3 classify.py      # rows.json -> classified.json, печатает покрытие
python3 gen.py           # -> ../categories.yaml, ../attributes.yaml, ../noise.yaml
```

- `cats.py` — категории: триггеры, анти-триггеры, синонимы.
- `attrspec.py` — характеристики по категориям и категории вне базы.
- `RESEARCH_OUT=/path/` — записать YAML в другую папку (для сравнения).

Правки yaml перенесены в скрипты; генератор отличается от текущих yaml только числами `строк_в_базе` у кабелей и проводов: они сняты до последней правки триггеров и не входят в знания, а yaml не перегенерирован, чтобы не менять версию знаний в дампе.
