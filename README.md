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

Семья трактора — 4 класса (`src/config/classes.py`, порядок фиксирован):

| Класс | Техника |
|-------|---------|
| `chtz` | ЧТЗ Б10М — гусеничный бульдозер |
| `johndeere` | John Deere — колёсный сельхозтрактор |
| `kirovets` | Кировец К-744 — тяжёлый колёсный трактор |
| `mtz_belarus` | семейство МТЗ (объединяет бывшие `mtz_82` и `mtz_1221`) |

Состояние — 2 класса: `clean`, `dirty` (`clean=0`, `dirty=1`).

## Метрики

| Метрика | Значение |
|---------|----------|
| Accuracy семьи трактора, val | **1.000** |
| Accuracy состояния clean/dirty, val | **0.942** |
| Dirty recall на реальной грязи (held-out `data/real_dirty_val`) | **0.906** (цель ≥ 0.90) |

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
  "timestamp": "2026-08-31T12:00:00"
}
```

Коды ошибок: `422` — недопустимое расширение / пустой / слишком большой файл;
`500` — модель не загружена или битое изображение.

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
