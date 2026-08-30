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
│   └── predict.py          predict_image(model, image, transform) -> (idx, conf, state_idx)
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
      ▼
PredictionResponse: model_class, confidence, state, processing_time, timestamp
```

Модель грузится один раз в `lifespan` FastAPI через
`resolve_working_checkpoint()` → `load_multi_task_model()`. Если рабочего
чекпоинта нет, сервис поднимается, `/health` возвращает `models_loaded: false`,
а `/predict` отвечает `500`.

## Расширение архитектуры

Заложено на будущее, пока не реализовано:

- **Детектор** первой стадии (кроп трактора до классификации) — вынесет фон и
  поднимет точность на «дальних» кадрах.
- **Головы подмоделей** внутри семьи (например, модификации МТЗ) — добавляются
  как ещё одна `Linear`-голова на том же эмбеддинге, по образцу `state_head`.
- **Feedback-петля**: ответы `/predict` с низкой `confidence` собираются в
  `to_review` и после ручной проверки вливаются в датасет.
