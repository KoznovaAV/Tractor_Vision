# Tractor Vision — правила для Claude Code

## Проект
Распознавание семей тракторов по фото + состояние (грязный/чистый).
FastAPI + PyTorch Lightning + ConvNeXt-Tiny (multi-task: 4 семьи + state).

## Окружение
- ОС: Windows 11, PowerShell
- Python: 3.11 в conda-окружении `tractor`
- GPU: NVIDIA RTX 4050 Laptop (CUDA)
- Все команды выполняются из корня проекта
- Ветки: работа ведётся в `refactor/arch`, не в `main`

## Жёсткие правила
1. Классы и размеры — ТОЛЬКО из `src/config/classes.py` и `config.yaml`. Хардкод запрещён.
2. Новые скрипты: `argparse` + type hints + docstring (Google-style, русский язык).
3. Перед коммитом: `pytest tests/ -q` зелёный, pre-commit хуки проходят.
4. НЕ удалять/переименовывать `data/*` и `weights/*` без явной команды человека.
5. Крупные правки — только в ветке `refactor/arch`.
6. Коммиты атомарные: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`.

## Карта кода
- `src/config/` — источник правды (classes.py, config_loader.py)
- `src/data/` — датасет, трансформы, деградации (dataset.py, transforms.py, generate_dirty_dataset.py)
- `src/data/utils.py` — общие утилиты конвейеров данных (compute_content_hash)
- `src/models/` — архитектуры (multi_task.py)
- `src/models/registry.py` — реестр моделей инференса из config.models
- `src/training/` — обучение/оценка (multi_task_train.py, multi_task_evaluate.py)
- `src/api/` — FastAPI (main.py, schemas.py)
- `scripts/` — живые конвейеры (collect, prepare, generate, real-dirty цикл, demo)
- `scripts/error_analysis_state.py` — разбор промахов головы состояния; `--sweep`
  калибрует `state_dirty_threshold` по dirty recall на `data/real_dirty_val`
- `scripts/ingest_feedback.py` — вливание пользовательского фидбэка в train
- `tests/` — тесты (64 шт., все на моках/`tmp_path`, не требуют `weights/` и `data/`)
- `docs/` — ARCHITECTURE, MODEL_CARD, DATA, DEPLOYMENT, RETRAIN
- `config.yaml` — единый конфиг
- `.github/workflows/ci.yml` — GitHub Actions: pre-commit + pytest на CPU-torch

## Текущее состояние (01.09.2026)
- 4 класса: `chtz`, `johndeere`, `kirovets`, `mtz_belarus` (без подмоделей)
- Family accuracy на val: 0.997
- Real dirty recall (`data/real_dirty_val`): 0.893 (28 фото)
- Probe false-dirty (`data/real_clean_probe`, argmax): 0.158
- `state_dirty_threshold`: 0.60 (откалиброван через `error_analysis_state.py --sweep`)
- API: `/health`, `/models` (реестр из config.models), `/predict`
  (request_id + needs_review + model_version/checkpoint_sha), `/predict_batch`
  (пофайловая обработка ошибок), `/feedback`
- Аутентификация: заголовок `X-API-Key`, ключи в env `TRACTOR_VISION_API_KEYS`
  (список через запятую); при пустом env auth отключена
- Rate limit: пер-ключ, in-memory окно (настройки в config.yaml)
- Обучение: `--init-checkpoint` (дообучение с готовых весов),
  `--early-stopping-patience` (ранняя остановка по val)
- Docker: CPU образ собран, но устарел (нужна пересборка)

## Стиль
- black/isort/flake8 по `setup.cfg`
- Комментарии — только «почему», не «что делает» (это видно из кода)
- История изменений живёт в git, а не в коде
- Редактируешь файл — обновляй его docstring: без истории, только текущее поведение.

## Выполнено в Спеке 11
- Удалён мёртвый single-task код (`classifier.py`, `train.py`, `evaluate.py`),
  одноразовые миграции (`merge_mtz1221.py`); `eval_visualize` слит в
  `multi_task_evaluate` (флаг `--out-dir`).
- Общие утилиты: загрузчик чекпоинта (`src/models/loader.py`) и инференс
  одного изображения (`src/models/predict.py`).
- Канонические имена классов в источниках данных; Docker CMD по умолчанию — API.
- README переписан под 4 класса и `image_size 384`; создан `docs/`
  (ARCHITECTURE, MODEL_CARD, DATA, DEPLOYMENT, RETRAIN); удалён
  `RETRAIN_CHECKLIST.md`.
- Вычищены исторические комментарии; docstring описывает только текущее поведение.
