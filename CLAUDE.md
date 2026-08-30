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
- `src/models/` — архитектуры (multi_task.py)
- `src/training/` — обучение/оценка (multi_task_train.py, multi_task_evaluate.py)
- `src/api/` — FastAPI (main.py, schemas.py)
- `scripts/` — живые конвейеры (collect, prepare, generate, real-dirty цикл, demo)
- `tests/` — тесты
- `docs/` — документация (пока пусто)
- `config.yaml` — единый конфиг

## Текущее состояние (31.08.2026)
- 4 класса: `chtz`, `johndeere`, `kirovets`, `mtz_belarus` (без подмоделей)
- Model accuracy на val: 1.000
- State accuracy на val: 0.942
- Dirty recall на реальной грязи: 0.906 (цель ≥ 0.90 достигнута)
- API: `/health`, `/models`, `/predict` (multi-task only)
- Docker: CPU образ собран, но устарел (нужна пересборка)

## Стиль
- black/isort/flake8 по `setup.cfg`
- Комментарии — только «почему», не «что делает» (это видно из кода)
- История изменений живёт в git, а не в коде
- Редактируешь файл — обновляй его docstring: без истории, только текущее поведение.

## Текущие задачи (архитектурная чистка)
1. Удалить мёртвый код:
   - `src/models/classifier.py`, `src/training/train.py`, `src/training/evaluate.py` (single-task цепочка)
   - `scripts/merge_mtz1221.py` (одноразовая миграция)
   - `scripts/eval_visualize.py` (слить в multi_task_evaluate с флагом `--out-dir`)
2. Почистить папки `data/` от рабочих лесов
3. Создать документацию: `docs/ARCHITECTURE.md`, `MODEL_CARD.md`, `DATA.md`, `DEPLOYMENT.md`, `RETRAIN.md`
4. Подготовить архитектуру к расширению (детали, подмодели через feedback)
5. Пересобрать Docker-образ (CPU) с актуальным кодом и новым чекпоинтом
6. Сохранить работоспособность: pytest зелёный, API работает, Docker запускается
