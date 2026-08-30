# Данные

Все датасеты живут в `data/` и **не** версионируются git (`/data/` в
`.gitignore`). Веса — в `weights/`, тоже вне git.

## Классы

4 семьи трактора: `chtz`, `johndeere`, `kirovets`, `mtz_belarus`. Имена и
порядок — из `src/config/classes.py`. Псевдонимы `mtz_82` и `mtz_1221` сводятся
к `mtz_belarus` (`CLASS_ALIASES`) при разметке и сборе.

Состояние: `clean`, `dirty` (порядок фиксирован: `clean=0`, `dirty=1`).

## Источники изображений

| Директория | Что | Как получено |
|------------|-----|--------------|
| `data/raw/<class>/` | исходные фото по классам, канонические имена | ручной сбор |
| `data/collected/<split>/<class>/` | автосбор из веба | `scripts/collect_tractors.py` (DuckDuckGo + phash-дедуп + CLIP-фильтр) |
| `data/real_dirty_raw/unsorted/` | реальные грязные фото без раскладки | `scripts/collect_dirty_tractors.py` |
| `data/real_dirty_labeled/<class>/` | грязные фото, разложенные по классам | `scripts/pseudo_label_dirty.py` (+ ручная проверка `to_review/`) |

Фото с низкой уверенностью классификатора попадают в `to_review/` и
раскладываются вручную — единственный обязательный ручной шаг.

## Рабочие деревья

| Директория | Структура | Назначение |
|------------|-----------|------------|
| `data/processed/` | `<split>/<class>/*.jpg` | чистый датасет, вход генератора грязи |
| `data/dirty_clean/` | `<split>/<class>/{clean,dirty}/*.jpg` | multi-task датасет обучения |
| `data/real_dirty_val/` | `<class>/*.jpg` (все `dirty`) | held-out оценка на реальной грязи |
| `data/real_dirty_reserve/` | `<class>/*.jpg` (все `dirty`) | резерв реальной грязи (см. ниже) |

Сплиты: `train` / `val` / `test`. И `data/processed`, и `data/dirty_clean`
содержат все три сплита. `data/*_backup/` — ручные резервные копии перед
пересборкой, в конвейере не используются.

## Разбиение

`scripts/prepare_dataset.py` объединяет источники и **стратифицированно**
нарезает сплиты в пропорции **70 / 15 / 15**:

- страта = `(класс, состояние)`, если есть уровень `clean/dirty`, иначе `класс`;
- дедупликация между источниками по MD5 содержимого — одно фото не попадёт
  одновременно в `train` и `test`;
- классы берутся явно из `MODEL_CLASSES`, служебные папки игнорируются;
- `--seed` фиксирует разбиение.

Генерация грязи (`src/data/generate_dirty_dataset.py`) работает **после**
пересборки: зеркалит `data/processed/<split>/<class>/` в
`data/dirty_clean/<split>/<class>/`, кладёт оригиналы в `clean/`, а к каждому
добавляет `--dirty-per-clean` (по умолчанию 2) синтетических грязных варианта в
`dirty/`. Деградации компонуемые (грязь, брызги, пыль, дождь, туман, перекрытия,
корка ила `mud_crust`, освещение), сила рандомизирована, минимум один «грязевой»
эффект гарантирован. Грязь смещена к низу кадра (техника), небо остаётся чистым.

## Цикл real-dirty (дообучение головы состояния)

Класс техники распознаётся отлично, но на реальной экстремальной грязи
(сплошная корка ила) голова состояния проседает, т.к. обучалась на синтетике.
Цикл добавляет в обучение реальные грязные фото:

```
scripts/collect_dirty_tractors.py     -> data/real_dirty_raw/unsorted/
scripts/pseudo_label_dirty.py         -> data/real_dirty_labeled/<class>/  (+ ручная проверка to_review/)
scripts/split_real_dirty.py           -> 80% в data/dirty_clean/train/<class>/dirty/ (префикс real_)
                                         20% в data/real_dirty_val/<class>/  (held-out)
scripts/eval_real_dirty.py            -> dirty recall на held-out (цель >= 0.90)
src/data/generate_dirty_dataset.py    -> регенерация синтетики (с mud_crust)
src/training/multi_task_train.py      -> дообучение
scripts/eval_real_dirty.py            -> сравнение с baseline
```

Префикс `real_` в именах файлов отличает реальные грязные фото от синтетических
(`*_dirty0`) внутри общей папки `dirty/`. Разбиение 80/20 стратифицировано по
классам и идемпотентно (имя = префикс + content-hash). Пошагово — `RETRAIN.md`.

### `data/real_dirty_val`

Held-out набор реальной грязи: 20% размеченных реальных грязных фото, **не**
участвуют в обучении. На нём `scripts/eval_real_dirty.py` меряет **dirty
recall** (доля грязных фото, распознанных как `dirty`) до и после дообучения.
Текущее значение — **0.906** при цели ≥ 0.90.

### `data/real_dirty_reserve`

Резервный пул реальных грязных фото, отложенный **вне** и обучения
(`data/dirty_clean`), и текущего held-out (`data/real_dirty_val`). Нужен, чтобы
при повторных итерациях дообучения оставался свежий, ни разу не виденный
моделью набор реальной грязи: без резерва каждый новый цикл постепенно вливает
весь собранный корпус в `train`, и честно оценивать обобщение становится не на
чем. Фото из резерва вводятся в оценку (или в `real_dirty_val`) осознанно,
отдельным решением, а не автоматически.

## Проверки консистентности

```powershell
# классы согласованы с конфигом
python -c "from src.config.classes import MODEL_CLASSES; from src.config.config_loader import load_config; assert load_config().num_model_classes == len(MODEL_CLASSES) == 4; print('OK', MODEL_CLASSES)"

# единый размер изображения
python -c "from src.config.config_loader import load_config; assert load_config().image_size == 384; print('image_size: 384')"
```

Глазами: грязь на технике, не на небе; `to_review/` вычищен до `prepare_dataset`;
матрица ошибок 4×4; классы представлены во всех трёх сплитах.
