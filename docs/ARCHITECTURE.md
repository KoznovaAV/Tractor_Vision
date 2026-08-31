# Архитектура

Tractor Vision — multi-task система распознавания: по одному фото определяет
семью трактора (4 класса) и состояние техники (`clean` / `dirty`) за один
проход одной сети.

## Стек

- **PyTorch + torchvision** — модель (ConvNeXt-Tiny).
- **PyTorch Lightning** — цикл обучения, чекпоинты, колбэки.
- **Albumentations** — аугментации и препроцессинг (train / val / инференс).
- **FastAPI + Uvicorn** — HTTP-сервис инференса.
- **Docker** — образ инференса и тестов (CPU по умолчанию, опционально CUDA).

## Модель

`src/models/multi_task.py` — `MultiTaskTractorClassifier`:

```
изображение (B, 3, 384, 384)
        │
        ▼
ConvNeXt-Tiny backbone (torchvision, веса ImageNet)
classifier заменён на nn.Identity
        │
        ▼
вектор признаков (B, 768)
        ├───────────────► model_head: Linear(768 → 4)   логиты семьи трактора
        └───────────────► state_head: Linear(768 → 2)   логиты состояния
```

Обе головы читают один и тот же эмбеддинг — backbone считается один раз.
Число классов приходит из `src/config/classes.py`, размер входа — из
`config.yaml`.

## Источник правды

| Что | Где |
|-----|-----|
| Имена и порядок классов, псевдонимы | `src/config/classes.py` |
| `image_size`, число классов, пути, параметры API, fallback-метрики | `config.yaml` |
| Типизированный доступ к конфигу, чтение accuracy из чекпоинта | `src/config/config_loader.py` |

Хардкод классов и размеров в остальном коде запрещён (правило проекта).

## Карта модулей

```
src/
├── config/
│   ├── classes.py          MODEL_CLASSES, STATE_CLASSES, class_to_idx, псевдонимы
│   └── config_loader.py    load_config() -> AppConfig; read_checkpoint_accuracy()
├── data/
│   ├── dataset.py          TractorDataset: классы итерируются явно из MODEL_CLASSES
│   ├── dataloader.py       get_dataloader(split, ...) -> DataLoader
│   ├── transforms.py       get_train_transforms / get_val_transforms (Albumentations)
│   └── generate_dirty_dataset.py  синтетическая грязь поверх чистых фото
├── models/
│   ├── multi_task.py       MultiTaskTractorClassifier
│   ├── loader.py           load_multi_task_model(), resolve_working_checkpoint()
│   ├── registry.py         build_registry(config), get_model(entry) — реестр из config.models
│   └── predict.py          predict_image(...) -> (idx, conf, state_idx, state_conf)
├── training/
│   ├── multi_task_train.py     partial fine-tuning + uncertainty / gradnorm
│   └── multi_task_evaluate.py  accuracy на val + опц. confusion matrix
└── api/
    ├── main.py             FastAPI: /health, /models, /predict
    └── schemas.py          Pydantic-схемы ответов
```

`scripts/` — конвейеры подготовки данных и цикла real-dirty (см. `DATA.md`).

## Поток данных: обучение

```
data/collected            сырые фото (коллектор)
      │  scripts/prepare_dataset.py  (объединение источников + стратиф. сплиты 70/15/15)
      ▼
data/processed            чистое дерево <split>/<class>/
      │  src/data/generate_dirty_dataset.py  (2 грязных варианта на чистое фото)
      ▼
data/dirty_clean          multi-task дерево <split>/<class>/{clean,dirty}/
      │  src/training/multi_task_train.py
      ▼
weights/multi-task-best-*.ckpt  +  weights/multi_task_best.ckpt (рабочий)
      │  src/training/multi_task_evaluate.py
      ▼
метрики val (model acc / state acc)
```

Цикл дообучения головы состояния на реальной грязи описан в `RETRAIN.md`.

## Поток данных: инференс

```
POST /predict (multipart file)
      │  _validate_upload: расширение из config.api.allowed_extensions
      │  проверка размера: config.api.max_file_size_mb
      ▼
PIL.Image -> RGB
      │  get_val_transforms(image_size)  Resize + Normalize + ToTensor
      ▼
MultiTaskTractorClassifier(tensor)  ->  (model_logits, state_logits)
      │  softmax + argmax  (src/models/predict.predict_image)
      │  состояние: p(dirty) >= config.api.state_dirty_threshold ? dirty : clean
      ▼
PredictionResponse: model_class, confidence, state, state_confidence,
                    processing_time, timestamp
```

## Реестр моделей

Набор моделей инференса объявляется разделом `models` в `config.yaml`: ключ —
имя модели в API (параметр `?model=`), значение — чекпоинт, тип загрузчика и
список задач. `src/models/registry.py` разбирает раздел в `dict[str,
ModelEntry]` через `build_registry(config)`; `ModelEntry` — неизменяемый
dataclass (`name`, `checkpoint`, `type`, `tasks`).

`get_model(entry)` загружает веса для записи и кэширует результат (`lru_cache` по
самой записи): повторный вызов с той же записью возвращает тот же объект модели
без повторного чтения чекпоинта. Поддерживается тип `multi_task`
(`load_multi_task_model()`); при отсутствии файла чекпоинта записи берётся
актуальный рабочий чекпоинт через `resolve_working_checkpoint()`, иначе —
`ValueError` / `FileNotFoundError`.

`lifespan` FastAPI при старте проходит по реестру и грузит каждую запись
независимо через `get_model(entry)`; записи без доступного чекпоинта или с
неподдерживаемым типом пропускаются. `/models` перечисляет загруженные записи
(accuracy — из метаданных чекпоинта, fallback — `config.fallback_accuracy` по
типу). `/predict` использует запись `machine` (константа `DEFAULT_MODEL`),
параметр `?model=` выбирает другую: неизвестное имя — `422`, известное но не
загруженное — `500`. Если не загрузилась ни одна запись, сервис поднимается,
`/health` возвращает `models_loaded: false`, а `/predict` отвечает `500`.

## Feedback-цикл

Замкнутая петля дообучения головы состояния на реальных пользовательских
данных:

```
POST /predict            ответ с request_id, confidence, needs_review
      │  строка дописывается в output/predictions.jsonl
      ▼
пользователь видит ошибку (или needs_review) и присылает исправление
      │
POST /feedback           file + user_family (+ опц. user_state, request_id)
      │  валидация семьи по MODEL_CLASSES, состояния по STATE_CLASSES
      ▼
data/feedback/<user_family>/<stem>.{jpg,json}   фото + JSON-манифест
      │  накопление N примеров (ориентир — 50, см. RETRAIN.md)
      ▼
python -m scripts.ingest_feedback --apply
      │  валидация + дедуп по content-hash против data/processed и data/dirty_clean
      │  состояние: из manifest.user_state, иначе прогноз multi-task моделью
      ▼
data/dirty_clean/train/<family>/<state>/feedback_*   вливание в обучающую выборку
      │  при необходимости — регенерация синтетики (mud_crust)
      ▼
переобучение по фазам RETRAIN.md  →  новый рабочий чекпоинт
```

`/feedback` только принимает и складывает данные — оно не трогает модель и не
меняет датасет. Всё вливание идёт офлайн через `scripts/ingest_feedback.py`
(по умолчанию сухой прогон, реальное копирование — `--apply`); фото и манифест
**копируются**, каталог `data/feedback` остаётся нетронутым. Подробности пула —
`DATA.md`, шаги переобучения — `RETRAIN.md`.

## Расширение архитектуры

Заложено на будущее, пока не реализовано:

- **Детектор** первой стадии (кроп трактора до классификации) — вынесет фон и
  поднимет точность на «дальних» кадрах.
- **Головы подмоделей** внутри семьи (например, модификации МТЗ) — добавляются
  как ещё одна `Linear`-голова на том же эмбеддинге, по образцу `state_head`.
- **Автотриггер переобучения**: сейчас накопление фидбэка и запуск
  `ingest_feedback` — ручной шаг (см. «Feedback-цикл»); порог по числу примеров
  можно повесить на планировщик.
