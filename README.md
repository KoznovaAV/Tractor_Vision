# 🚜 Tractor Vision

Multi-task компьютерное зрение для аудита парка техники: по одному фото
определяет **семью трактора** и **состояние** (`clean` / `dirty`) за один
проход одной сети.

- **Backbone:** ConvNeXt-Tiny (torchvision, предобучен на ImageNet), две
  линейные головы над общим эмбеддингом.
- **Стек:** PyTorch + PyTorch Lightning, Albumentations, FastAPI, Docker.
- **Вход:** RGB, `image_size = 384` (единый для train / eval / инференса).

Документация: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/DATA.md](docs/DATA.md) ·
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) · [docs/RETRAIN.md](docs/RETRAIN.md)

## Классы

Модель распознаёт 4 семейства тракторов. Каждый класс — это всё семейство,
а не конкретная модель; подмодели не различаются (уровень 2 отключён).

| Класс | Семейство | Примеры моделей семейства |
|-------|-----------|---------------------------|
| `chtz` | ЧТЗ (Уралтрак) | Б10М, Т-170, Т-130 |
| `johndeere` | John Deere | 6R, 7R, 8R |
| `kirovets` | Кировец | К-744, К-7М, К-525 |
| `mtz_belarus` | МТЗ / Belarus | 82, 1221, 1523, 2022 |

Каждый класс — семейство техники целиком, без деления на подмодели и
модификации: `mtz_belarus` покрывает весь модельный ряд МТЗ одной меткой.

Состояние — 2 класса: `clean`, `dirty` (`clean=0`, `dirty=1`).

## Метрики

| Метрика | Значение |
|---------|----------|
| Accuracy семьи трактора, val | **1.000** |
| Accuracy состояния clean/dirty, val | **0.942** |
| Dirty recall на реальной грязи (held-out `data/real_dirty_val`) | **0.906** (цель ≥ 0.90) |
| Порог `needs_review` в `/predict` | `confidence < 0.6` → ответ помечается на ручную проверку |

Val-метрики — на синтетической грязи `data/dirty_clean/val`. Подробности,
ограничения и версия чекпоинта — [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Быстрый старт

Требуется conda и Python 3.11. Окружение проекта — `tractor`.

```powershell
conda activate tractor
pip install -r requirements.txt
pre-commit install
```

### Тесты

```powershell
pytest tests/ -q
pre-commit run --all-files
```

### API локально

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: <http://localhost:8000/docs>. Для работающего `/predict` нужен
чекпоинт `weights/multi_task_best.ckpt`; без него сервис поднимется, но
`/health` вернёт `models_loaded: false`.

### Docker

```powershell
docker compose up --build -d api      # API на http://localhost:8000
docker compose run --rm test          # тесты в контейнере
```

CPU-образ по умолчанию; GPU-вариант —
`docker build --build-arg TORCH_VARIANT=cu121 -t tractor-vision:gpu .`
(см. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).

## API

Базовый URL: `http://localhost:8000`. Схемы ответов — `src/api/schemas.py`.

### `GET /health`

```json
{ "status": "healthy", "version": "1.0.0", "models_loaded": true }
```

### `GET /models`

Список загруженных моделей; `accuracy` читается из метаданных чекпоинта, при
отсутствии — из `config.yaml:fallback_accuracy`.

```json
{
  "models": [
    { "name": "Multi-Task Classifier", "num_classes": 4, "accuracy": 1.0,
      "weights_path": "weights/multi_task_best.ckpt" }
  ],
  "count": 1
}
```

### `POST /predict`

`multipart/form-data`, поле `file` — изображение JPEG/PNG (лимит из
`config.yaml:api.max_file_size_mb`, по умолчанию 10 МБ).

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@tractor.jpg;type=image/jpeg"
```

```json
{
  "model_class": "mtz_belarus",
  "confidence": 0.987,
  "state": "clean",
  "processing_time": 0.123,
  "timestamp": "2026-08-31T12:00:00",
  "request_id": "3f2a1c9e-8b7d-4e6a-9f10-2c5d8a1b3e4f",
  "needs_review": false
}
```

`needs_review` — `true`, когда `confidence` ниже `config.yaml:api.confidence_threshold`
(по умолчанию `0.6`): такой ответ стоит перепроверить и при ошибке отправить в
`/feedback`. Каждая строка предсказания дописывается в `output/predictions.jsonl`.
Параметр `?model=` выбирает модель из реестра (по умолчанию `machine`).

Коды ошибок: `422` — недопустимое расширение / пустой / слишком большой файл /
неизвестная модель; `500` — модель не загружена или битое изображение.

### `POST /feedback`

Исправление пользователя: `multipart/form-data` с полем `file` (изображение) и
`user_family` (правильная семья, валидируется по `MODEL_CLASSES`). Опционально —
`user_state` (`clean`/`dirty`) и `request_id` исходного `/predict`. Фото и
JSON-манифест складываются в `config.yaml:api.feedback_dir/<user_family>/`;
позже `scripts/ingest_feedback.py` вливает их в обучающую выборку (см.
[docs/DATA.md](docs/DATA.md), [docs/RETRAIN.md](docs/RETRAIN.md)).

```bash
curl -X POST http://localhost:8000/feedback \
  -F "file=@tractor.jpg;type=image/jpeg" \
  -F "user_family=kirovets" \
  -F "user_state=dirty" \
  -F "request_id=3f2a1c9e-8b7d-4e6a-9f10-2c5d8a1b3e4f"
```

```json
{ "saved": true, "path": "data/feedback/kirovets/3f2a1c9e-8b7d-4e6a-9f10-2c5d8a1b3e4f.jpg" }
```

Коды ошибок: `422` — недопустимое расширение, неизвестная семья или состояние,
пустой / слишком большой файл.

## Как добавить класс техники

Классы объявлены **только** в `src/config/classes.py` — датасет, модель, API и
тесты берут их оттуда.

1. Сложить фото нового класса в `data/raw/<new_class>/` и провести через
   конвейер подготовки данных ([docs/DATA.md](docs/DATA.md)).
2. В `src/config/classes.py` добавить имя в конец кортежа `MODEL_CLASSES`
   (**в конец** — порядок является контрактом с весами головы). При необходимости
   добавить псевдонимы в `CLASS_ALIASES`.
3. В `config.yaml` увеличить `num_model_classes` до новой длины `MODEL_CLASSES`.
4. Переобучить модель — голова `model_head` инициализируется под новое число
   классов ([docs/RETRAIN.md](docs/RETRAIN.md)).
5. Тесты не хардкодят длину списка классов и подстроятся автоматически;
   запустить `pytest tests/ -q`.

Размер входа и прочие параметры менять там же — в `config.yaml`. Хардкод
классов и размеров в коде запрещён.

## Структура репозитория

```
Tractor_Vision/
├── config.yaml                 единый конфиг (размеры, классы, пути, API)
├── src/
│   ├── config/                 classes.py, config_loader.py — источник правды
│   ├── data/                   dataset, dataloader, transforms, generate_dirty_dataset
│   ├── models/                 multi_task (архитектура), loader, predict
│   ├── training/               multi_task_train, multi_task_evaluate
│   └── api/                    main.py (FastAPI), schemas.py
├── scripts/                    конвейеры сбора/подготовки данных, цикл real-dirty, демо
├── tests/                      pytest: API, датасет, data pipeline
├── docs/                       ARCHITECTURE, MODEL_CARD, DATA, DEPLOYMENT, RETRAIN
├── weights/                    чекпоинты (вне git)
├── data/                       датасеты (вне git)
├── Dockerfile                  образ инференса/тестов (CPU по умолчанию, опц. CUDA)
├── docker-compose.yml          сервисы api и test
├── requirements.txt            зависимости для разработки и обучения
├── requirements-docker.txt     пиновка зависимостей для образа
├── .pre-commit-config.yaml     black / isort / flake8
└── setup.cfg                   конфиг линтеров
```

## Автор

Кознова Алина Владимировна — Государственный университет «Дубна», ИСАУ,
Computer Science (Deep Learning Engineering / MLOps).
