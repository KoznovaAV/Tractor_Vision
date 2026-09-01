# Деплой

Docker-образ обслуживает инференс-API и прогон тестов. Обучение в Docker не
выполняется (идёт локально в conda-окружении `tractor`).

## Образ

`Dockerfile` — база `python:3.11-slim`. Вариант PyTorch выбирается build-arg
`TORCH_VARIANT`:

| Значение | Колёса | Применение |
|----------|--------|------------|
| `cpu` (по умолчанию) | `download.pytorch.org/whl/cpu` | инференс на любой машине, CI, тесты |
| `cu121` | `download.pytorch.org/whl/cu121` | GPU-инференс (CUDA 12.1) |

```powershell
# CPU
docker build -t tractor-vision:cpu .

# GPU (CUDA 12.1)
docker build --build-arg TORCH_VARIANT=cu121 -t tractor-vision:gpu .
```

В образ копируются `src/`, `scripts/`, `tests/`, `config.yaml` и создаются
пустые `weights/`, `data/`, `output/`. Веса и данные внутрь **не** зашиты —
подаются томами.

## Запуск через docker compose

`docker-compose.yml` описывает два сервиса, оба на CPU-образе `tractor-vision:cpu`:

| Сервис | Команда | Тома |
|--------|---------|------|
| `api` | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` | `./weights:ro`, `./config.yaml:ro` |
| `test` | `pytest tests/ -v --cov=src` | `./data`, `./weights` |

```powershell
docker compose up --build -d api      # API на http://localhost:8000
docker compose run --rm test          # тесты в контейнере
docker compose logs -f api
docker compose down
```

`api` имеет healthcheck (`GET /health` каждые 30 с, `start_period` 40 с) и
`restart: unless-stopped`.

## GPU через compose

Anchor `x-build-cpu` фиксирует `TORCH_VARIANT: cpu`. Для GPU:

1. собрать `tractor-vision:gpu` вручную (см. выше);
2. в сервисе `api` заменить `build` на `image: tractor-vision:gpu` и добавить
   `deploy.resources.reservations.devices` с `capabilities: [gpu]` (нужен
   NVIDIA Container Toolkit на хосте).

## Тома и файлы

| Путь в контейнере | Источник | Режим | Зачем |
|-------------------|----------|-------|-------|
| `/app/weights` | `./weights` | ro (api) / rw (test) | чекпоинт `multi_task_best.ckpt` |
| `/app/config.yaml` | `./config.yaml` | ro | размеры, классы, параметры API |
| `/app/data` | `./data` | rw | только для тестов, читающих `data/processed` |

Минимум для работающего `/predict` — смонтированный `weights/` с рабочим
чекпоинтом. Без него сервис поднимется, но `/health` вернёт
`models_loaded: false`, а `/predict` — `500`.

## Аутентификация и лимиты

Защита эндпоинтов управляется секцией `api` в `config.yaml`:

| Параметр | По умолчанию | Смысл |
|----------|--------------|-------|
| `api.auth_enabled` | `false` | требовать заголовок `X-API-Key` на `/models`, `/predict`, `/predict_batch`, `/feedback` |
| `api.rate_limit_rpm` | `60` | лимит запросов в минуту (скользящее окно 60 с, in-memory) |

`/health` и Swagger (`/docs`) открыты всегда.

Сами ключи в конфиг **не** попадают — они задаются переменной окружения
`TRACTOR_VISION_API_KEYS` (список через запятую). Если `auth_enabled: true`, а
переменная пуста, сервис не стартует с явной ошибкой.

Лимит частоты считается на ключ при включённой аутентификации и на IP клиента
(`request.client.host`) при выключенной. Превышение — ответ `429` с заголовком
`Retry-After` (секунды до освобождения слота).

```bash
# локальный запуск с аутентификацией
export TRACTOR_VISION_API_KEYS="prod-key-1,prod-key-2"
# в config.yaml: api.auth_enabled: true
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

curl -H "X-API-Key: prod-key-1" http://localhost:8000/models
```

Для `docker compose` ключи передаются через `environment` сервиса `api`
(значение подставляется из окружения хоста или `.env`):

```yaml
services:
  api:
    environment:
      TRACTOR_VISION_API_KEYS: ${TRACTOR_VISION_API_KEYS}
```

При этом `api.auth_enabled: true` включается в смонтированном `config.yaml`.

## Проверки после деплоя

```powershell
# сервис жив
curl http://localhost:8000/health
# -> {"status":"healthy","version":"1.0.0","models_loaded":true}

# модель и метрика загружены
curl http://localhost:8000/models
# -> models[0].accuracy прочитан из метаданных чекпоинта / fallback конфига

# инференс
curl -X POST http://localhost:8000/predict -F "file=@sample.jpg;type=image/jpeg"
# -> model_class, confidence, state, processing_time, timestamp

# Swagger
# http://localhost:8000/docs
```

`models_loaded: true` и осмысленный ответ `/predict` на реальном фото —
достаточный признак корректного деплоя.

## CI

`.github/workflows/ci.yml` на `push`/`PR` в `main`: pre-commit (не блокирует),
`pytest --cov`, затем сборка образа и прогон `pytest` внутри контейнера.
