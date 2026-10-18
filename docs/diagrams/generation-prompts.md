# Промпты схем архитектуры

Созданы встроенным image_gen. Визуальный референс — `02-search-flow.png`.

## 01-data-preparation.png

Create a NEW Russian infographic, using attached image ONLY as visual style reference (white, bold dark navy Cyrillic, light blue rounded blocks, crisp arrows, purple AI, subtle gradients). It is diagram 1 in the same series. Title «Подготовка базы к поиску». Compact portrait format, large readable text, no excessive blank space. Exact flow:
Yellow database cylinder «История закупок» / «2 381 строка».
Down arrow to blue block «1. Нормализация и категория» / «7*1.5 → 7×1,5» / «Определить категорию по ключевым словам».
Small blue caption «2 264 строки с категорией». Side branch from this block to gray terminal «Перевозка и погрузка» / «108 строк — не участвуют в поиске». These are separate statistics, do NOT subtract 108 from 2264 or invent totals.
Main down arrow to blue block «2. Извлечь характеристики» / «Правила читают название» / «7 жил · 1,5 мм² · нг-LS» / «937 строк».
Then split: straight arrow label «Распознано» bypasses optional AI and goes to final database; side dashed arrow label «Правил не хватило» to purple block «3. ИИ дочитывает характеристики» / «Выбирает из допустимых значений» / «Ещё 472 строки». AI arrow merges into final database.
Final yellow database cylinder «База для поиска» / «Категория · характеристики · эмбеддинг».
Small bottom caption «Подготавливается заранее для всей базы».
Do not show embeddings as produced by reading AI; simply include as stored field. Numbers historical from supplied architecture, preserve exactly. Avoid extra boxes, English, code, decorative imagery. Draw clear direction and bypass without crossing connectors. Text must be crisp.

## 03-knowledge-and-feedback.png

Create a NEW Russian infographic, attached image is ONLY style reference: white background, navy bold Cyrillic, blue process boxes, yellow data, green people, gray dashed planned work, clean rounded rectangles and arrows, subtle gradients. Diagram 3 of same series. Title «Кто и что меняет в поиске». Wide landscape or square composition with THREE horizontal lanes stacked vertically. Each lane reads left-to-right: source/person → data → impact. Large readable labels, no crossing arrows, generous spacing. Lane headings and exact content:
LANE 1 heading «Справочник — из стандартов». Left yellow block «КСР · ОКПД2 · ETIM», arrow to middle yellow database «Категории и характеристики», arrow to right blue block «Подготовка базы и разбор запроса» smaller «Шаги 1, 2 и 4». Small separate gray DASHED block beneath source «КТРУ — данные ещё не подключены», dashed arrow to middle database. This is future work, visually explicit.
LANE 2 heading «Правила замен — от специалиста». Left green block «Специалист на /knowledge» smaller «Проверяет влияние до публикации», arrow to middle yellow database «Правила сравнения» smaller «Обязательность · допуск · ±ступень», arrow to right blue block «Проверка кандидатов» smaller «Шаг 6».
LANE 3 heading «Золотые карточки — опыт специалиста». Left green block «Специалист оценивает выдачу» smaller «Та же · аналог · не подходит» and «Шаг 9», arrow to middle yellow database «Золотые карточки» smaller «Эталон из КСР, если есть», arrow to right blue block «Решения в следующем поиске» smaller «Шаг 7», and an additional small output connected from database «Контрольный набор для оценки качества».
Bottom short legend with small colored swatches «Данные и справочник» yellow, «Специалист» green, «Алгоритм» blue, «В планах» gray dashed.
Do not link lanes sequentially; these are THREE distinct mechanisms. No invented steps or statistics. Crisp accurate Cyrillic, same professional aesthetic as reference.
