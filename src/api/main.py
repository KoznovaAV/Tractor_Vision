#!/usr/bin/env python3
"""FastAPI-приложение инференса Tractor Vision.

Сервис поднимает единственную multi-task модель (семья трактора + состояние) и
отдаёт эндпоинты ``/health``, ``/models``, ``/predict``, ``/predict_batch``,
``/feedback``.

Каждый ответ ``/predict`` получает ``request_id`` (uuid4), флаг ``needs_review``
(уверенность ниже ``api.confidence_threshold``), ``state_confidence`` —
уверенность в состоянии, решённом по порогу ``api.state_dirty_threshold``
(``p(dirty)`` не ниже порога -> ``dirty``) — и идентификаторы модели
``model_version`` (метка из конфига) и ``checkpoint_sha`` (12 hex-символов
SHA-256 чекпоинта); строка предсказания с теми же полями дописывается в
``output/predictions.jsonl``. ``/predict_batch`` принимает список файлов (поле
``files``) и обрабатывает каждый независимо: сбой одного файла не роняет батч, а
даёт ``status="error"`` для этого элемента. ``/health`` перечисляет модели
реестра с версиями. ``/feedback`` принимает присланное пользователем фото с
исправленной семьёй (и опционально состоянием) и складывает его в
``api.feedback_dir/<user_family>/`` рядом с JSON-манифестом.

Набор моделей задаётся разделом ``models`` в ``config.yaml`` и разбирается в
реестр (:mod:`src.models.registry`): ``lifespan`` грузит по записи реестра каждую
доступную модель, ``/models`` перечисляет реестр, ``/predict`` использует запись
``machine`` (переопределяется параметром ``?model=``).

При ``api.auth_enabled: true`` эндпоинты ``/models``, ``/predict``,
``/predict_batch`` и ``/feedback`` требуют заголовок ``X-API-Key`` (ключи — из
переменной окружения ``TRACTOR_VISION_API_KEYS``, список через запятую; при
пустой переменной сервис не стартует). ``/health`` и Swagger открыты всегда. Ко
всем защищённым эндпоинтам применяется лимит частоты ``api.rate_limit_rpm``
(скользящее окно 60 с, in-memory): на ключ при включённой аутентификации, иначе
на IP клиента; превышение — ответ 429 с заголовком ``Retry-After``.

Параметры (``image_size``, пути к весам, ``version``, ``max_file_size``,
``max_batch_size``) берутся
из ``config.yaml`` через :mod:`src.config.config_loader`; accuracy модели — из
метаданных чекпоинта через :func:`resolve_accuracy` с fallback из конфига. Имена
классов — из :mod:`src.config.classes`. Инференс — общая утилита из
:mod:`src.models.predict`.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from PIL import Image

from src.api.schemas import (
    BatchItemResult,
    BatchPredictionResponse,
    FeedbackResponse,
    HealthModelVersion,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)
from src.config.classes import MODEL_CLASSES, STATE_CLASSES, state_to_idx
from src.config.config_loader import load_config, resolve_accuracy
from src.data.transforms import get_val_transforms
from src.models.multi_task import MultiTaskTractorClassifier
from src.models.predict import predict_image
from src.models.registry import build_registry, get_model, get_model_meta

# Конфигурация и реестр моделей читаются один раз при импорте модуля.
_CONFIG = load_config()
_REGISTRY = build_registry(_CONFIG)

# Модель по умолчанию для /predict, если параметр ?model= не задан.
DEFAULT_MODEL: str = "machine"

IMAGE_SIZE: int = _CONFIG.image_size
MAX_FILE_SIZE: int = _CONFIG.api.max_file_size_bytes
MAX_BATCH_SIZE: int = _CONFIG.api.max_batch_size
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(_CONFIG.api.allowed_extensions)
API_VERSION: str = _CONFIG.api.version
CONFIDENCE_THRESHOLD: float = _CONFIG.api.confidence_threshold
STATE_DIRTY_THRESHOLD: float = _CONFIG.api.state_dirty_threshold
FEEDBACK_DIR: Path = _CONFIG.api.feedback_dir
AUTH_ENABLED: bool = _CONFIG.api.auth_enabled
RATE_LIMIT_RPM: int = _CONFIG.api.rate_limit_rpm

# Ключи API не хранятся в конфиге: список через запятую в переменной окружения.
API_KEYS_ENV: str = "TRACTOR_VISION_API_KEYS"

# Ширина окна лимита частоты (сек).
_RATE_WINDOW_SEC: float = 60.0

# Идентификатор клиента (ключ или IP) -> монотонные метки запросов в окне.
_rate_calls: dict[str, list[float]] = defaultdict(list)

# Журнал предсказаний: одна JSON-строка на запрос /predict.
PREDICTIONS_LOG: Path = Path("output/predictions.jsonl")

# Валидационная трансформация на едином размере.
transform = get_val_transforms(IMAGE_SIZE)

# Модели, загруженные при старте: имя записи реестра -> модель в режиме eval.
_loaded_models: dict[str, MultiTaskTractorClassifier] = {}


def _configured_api_keys() -> frozenset[str]:
    """Разобрать ключи API из переменной окружения ``TRACTOR_VISION_API_KEYS``.

    Returns:
        Множество непустых ключей (список через запятую), либо пустое множество.
    """
    raw = os.getenv(API_KEYS_ENV, "")
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def _enforce_rate_limit(identity: str) -> None:
    """Ограничить частоту запросов идентификатора скользящим окном 60 секунд.

    Args:
        identity: Идентификатор клиента (API-ключ или IP).

    Raises:
        HTTPException: 429 с заголовком ``Retry-After`` (секунды до освобождения
            слота), если в окне уже ``RATE_LIMIT_RPM`` запросов.
    """
    now = time.monotonic()
    window_start = now - _RATE_WINDOW_SEC
    calls = _rate_calls[identity]
    calls[:] = [ts for ts in calls if ts > window_start]
    if len(calls) >= RATE_LIMIT_RPM:
        retry_after = max(1, math.ceil(calls[0] + _RATE_WINDOW_SEC - now))
        raise HTTPException(
            status_code=429,
            detail="Превышен лимит запросов. Повторите позже.",
            headers={"Retry-After": str(retry_after)},
        )
    calls.append(now)


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> None:
    """Зависимость защищённых эндпоинтов: проверка ключа и лимита частоты.

    При ``AUTH_ENABLED`` требует валидный заголовок ``X-API-Key`` и считает лимит
    частоты на ключ. Иначе ключ не нужен, а лимит считается на IP клиента
    (``request.client.host``). ``/health`` и Swagger зависимость не подключают.

    Args:
        request: Входящий запрос (нужен для IP клиента).
        x_api_key: Значение заголовка ``X-API-Key`` (может отсутствовать).

    Raises:
        HTTPException: 401 при отсутствующем или неверном ключе; 429 при
            превышении лимита частоты.
    """
    if AUTH_ENABLED:
        if x_api_key is None or x_api_key not in _configured_api_keys():
            raise HTTPException(status_code=401, detail="Отсутствует или неверный API-ключ.")
        identity = f"key:{x_api_key}"
    else:
        client = request.client
        identity = f"ip:{client.host if client else 'unknown'}"
    _enforce_rate_limit(identity)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Проверить конфигурацию аутентификации и загрузить модели реестра.

    Если ``auth_enabled: true``, но переменная окружения с ключами пуста —
    поднимается :class:`RuntimeError` (fail fast). Каждая запись реестра затем
    грузится независимо; записи без доступного чекпоинта или с неподдерживаемым
    типом пропускаются.

    Args:
        app: Экземпляр FastAPI.

    Yields:
        Управление на время жизни приложения.

    Raises:
        RuntimeError: При ``auth_enabled: true`` и пустой переменной окружения
            ``TRACTOR_VISION_API_KEYS``.
    """
    if AUTH_ENABLED and not _configured_api_keys():
        raise RuntimeError(
            f"auth_enabled=true, но переменная окружения {API_KEYS_ENV} пуста. "
            f"Задайте список ключей через запятую, например "
            f"{API_KEYS_ENV}=key-a,key-b."
        )

    for name, entry in _REGISTRY.items():
        try:
            _loaded_models[name] = get_model(entry)
        except (FileNotFoundError, ValueError):
            continue
    yield
    _loaded_models.clear()
    get_model.cache_clear()


app = FastAPI(title="Tractor Vision API", version=API_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Проверка здоровья сервиса.

    Returns:
        Статус сервиса, версия, флаг загруженности моделей и список моделей
        реестра с их метками версий.
    """
    models_loaded = bool(_loaded_models)
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=models_loaded,
        models=[
            HealthModelVersion(name=entry.name, version=entry.version)
            for entry in _REGISTRY.values()
        ],
    )


@app.get("/models", dependencies=[Depends(require_api_key)])
async def list_models() -> dict[str, Any]:
    """Список моделей реестра, загруженных при старте, с их метриками.

    Accuracy читается из метаданных чекпоинта записи, иначе из fallback конфига
    по типу модели.

    Returns:
        Словарь с ключами ``models`` (список) и ``count`` (число).
    """
    models: list[ModelInfo] = []

    for name, entry in _REGISTRY.items():
        if name not in _loaded_models:
            continue
        accuracy = resolve_accuracy(
            entry.checkpoint,
            _CONFIG.fallback_accuracy.get(entry.type),
        )
        models.append(
            ModelInfo(
                name=entry.name,
                num_classes=len(MODEL_CLASSES),
                accuracy=accuracy if accuracy is not None else 0.0,
                weights_path=str(entry.checkpoint),
            )
        )

    return {
        "models": [m.model_dump() for m in models],
        "count": len(models),
    }


def _validate_upload(file: UploadFile) -> None:
    """Проверить расширение загружаемого файла.

    Args:
        file: Загружаемый файл.

    Raises:
        HTTPException: 422, если расширение не разрешено.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Недопустимое расширение: {suffix!r}. "
            f"Разрешены: {sorted(ALLOWED_EXTENSIONS)}.",
        )


def _decide_state(state_idx: int, state_conf: float) -> tuple[str, float]:
    """Решить состояние по порогу ``api.state_dirty_threshold``.

    Голова состояния бинарна (clean/dirty), поэтому ``p(dirty)`` восстанавливается
    из уверенности argmax-состояния. Состояние — ``dirty``, если ``p(dirty)`` не
    ниже порога, иначе ``clean``.

    Args:
        state_idx: Индекс состояния по argmax из ``predict_image``.
        state_conf: Softmax-уверенность этого состояния.

    Returns:
        Кортеж ``(state, state_confidence)``: имя решённого состояния и его
        уверенность (``p(dirty)`` для ``dirty``, ``1 - p(dirty)`` для ``clean``).
    """
    dirty_idx = state_to_idx("dirty")
    p_dirty = state_conf if state_idx == dirty_idx else 1.0 - state_conf
    if p_dirty >= STATE_DIRTY_THRESHOLD:
        return "dirty", p_dirty
    return "clean", 1.0 - p_dirty


def _run_inference(
    image: Image.Image, model_name: str
) -> tuple[str, float, str, float, str | None, str]:
    """Выполнить инференс моделью реестра по имени.

    Args:
        image: Открытое изображение ``PIL.Image``.
        model_name: Имя записи реестра (параметр ``?model=``).

    Returns:
        Кортеж ``(model_class, confidence, state, state_confidence,
        model_version, checkpoint_sha)``.

    Raises:
        HTTPException: 422, если модель не значится в реестре; 500, если модель
            есть в реестре, но не загружена.
    """
    if model_name not in _REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Неизвестная модель: {model_name!r}. Доступны: {sorted(_REGISTRY)}.",
        )

    if model_name not in _loaded_models:
        raise HTTPException(status_code=500, detail="No models loaded")

    model, model_version, checkpoint_sha = get_model_meta(_REGISTRY[model_name])
    model_idx, confidence, state_idx, state_conf = predict_image(model, image, transform)
    state, state_confidence = _decide_state(state_idx, state_conf)
    return (
        MODEL_CLASSES[model_idx],
        confidence,
        state,
        state_confidence,
        model_version,
        checkpoint_sha,
    )


def _log_prediction(
    request_id: str,
    family: str,
    state: str,
    confidence: float,
    state_confidence: float,
    model_version: str | None,
    checkpoint_sha: str,
) -> None:
    """Дописать строку предсказания в ``output/predictions.jsonl``.

    Args:
        request_id: Идентификатор запроса ``/predict``.
        family: Предсказанная семья трактора.
        state: Предсказанное состояние (clean/dirty).
        confidence: Уверенность модели в семье.
        state_confidence: Уверенность в решённом состоянии.
        model_version: Метка версии модели из конфига (``None``, если не задана).
        checkpoint_sha: Первые 12 hex-символов SHA-256 файла чекпоинта.
    """
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "request_id": request_id,
        "ts": datetime.now().isoformat(),
        "family": family,
        "state": state,
        "confidence": confidence,
        "state_confidence": state_confidence,
        "model_version": model_version,
        "checkpoint_sha": checkpoint_sha,
    }
    with PREDICTIONS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _predict_contents(contents: bytes, model_name: str) -> PredictionResponse:
    """Обработать байты одного изображения: инференс, лог, ответ.

    Общее ядро ``/predict`` и ``/predict_batch``. Пишет строку предсказания в
    ``output/predictions.jsonl`` со своим ``request_id``.

    Args:
        contents: Сырые байты изображения.
        model_name: Имя модели из реестра.

    Returns:
        Предсказание для этого изображения.

    Raises:
        HTTPException: 422 при пустом/слишком большом файле или неизвестной
            модели; 500 при отсутствии моделей или ошибке декодирования
            изображения.
    """
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Пустой файл.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Файл превышает лимит {MAX_FILE_SIZE} байт.",
        )

    start = time.perf_counter()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {exc}") from exc

    (
        model_class,
        confidence,
        state,
        state_confidence,
        model_version,
        checkpoint_sha,
    ) = _run_inference(image, model_name)
    processing_time = time.perf_counter() - start

    request_id = str(uuid4())
    needs_review = confidence < CONFIDENCE_THRESHOLD
    _log_prediction(
        request_id,
        model_class,
        state,
        confidence,
        state_confidence,
        model_version,
        checkpoint_sha,
    )

    return PredictionResponse(
        model_class=model_class,
        confidence=confidence,
        state=state,
        state_confidence=state_confidence,
        processing_time=processing_time,
        timestamp=datetime.now(),
        request_id=request_id,
        needs_review=needs_review,
        model_version=model_version,
        checkpoint_sha=checkpoint_sha,
    )


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
async def predict(
    file: UploadFile = File(...),
    model: str = Query(default=DEFAULT_MODEL),
) -> PredictionResponse:
    """Классифицировать трактор по изображению.

    Args:
        file: Загружаемое изображение (JPEG/PNG, максимум из конфига).
        model: Имя модели из реестра; по умолчанию ``machine``.

    Returns:
        Предсказание с классом модели, уверенностью, состоянием, таймингом,
        идентификатором запроса и флагом ``needs_review``.

    Raises:
        HTTPException: 422 при невалидном/пустом/слишком большом файле или
            неизвестной модели; 500 при отсутствии моделей или ошибке обработки
            изображения.
    """
    _validate_upload(file)
    contents = await file.read()
    return _predict_contents(contents, model)


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    dependencies=[Depends(require_api_key)],
)
async def predict_batch(
    files: list[UploadFile] = File(...),
    model: str = Query(default=DEFAULT_MODEL),
) -> BatchPredictionResponse:
    """Классифицировать пачку изображений одним запросом.

    Каждый файл обрабатывается независимо: успех даёт ``status="ok"`` с полем
    ``prediction`` (и строкой в ``output/predictions.jsonl`` со своим
    ``request_id``, как в ``/predict``), ошибка — ``status="error"`` с текстом
    причины. Ответ имеет код 200, если батч в принципе обработан, даже при
    частичных ошибках.

    Args:
        files: Список загружаемых изображений (поле ``files`` multipart-формы).
        model: Имя модели из реестра; по умолчанию ``machine``.

    Returns:
        Список результатов по файлам со счётчиками ``processed`` и ``failed``.

    Raises:
        HTTPException: 422, если не передано ни одного файла; 413, если файлов
            больше ``api.max_batch_size``.
    """
    if len(files) == 0:
        raise HTTPException(status_code=422, detail="Пустой батч: не передано ни одного файла.")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Размер батча {len(files)} превышает лимит {MAX_BATCH_SIZE}.",
        )

    results: list[BatchItemResult] = []
    for file in files:
        file_name = file.filename or ""
        try:
            _validate_upload(file)
            contents = await file.read()
            prediction = _predict_contents(contents, model)
        except HTTPException as exc:
            results.append(
                BatchItemResult(file_name=file_name, status="error", error=str(exc.detail))
            )
        except Exception as exc:  # noqa: BLE001 - любой сбой файла не должен ронять батч
            results.append(BatchItemResult(file_name=file_name, status="error", error=str(exc)))
        else:
            results.append(BatchItemResult(file_name=file_name, status="ok", prediction=prediction))

    ok = [r for r in results if r.status == "ok"]
    first_ok = ok[0].prediction if ok else None
    return BatchPredictionResponse(
        results=results,
        processed=len(ok),
        failed=len(results) - len(ok),
        model_version=first_ok.model_version if first_ok else None,
        checkpoint_sha=first_ok.checkpoint_sha if first_ok else None,
    )


@app.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(require_api_key)])
async def feedback(
    file: UploadFile = File(...),
    user_family: str = Form(...),
    request_id: str | None = Form(default=None),
    user_state: str | None = Form(default=None),
) -> FeedbackResponse:
    """Принять исправление пользователя и сохранить фото с манифестом.

    Фото кладётся в ``api.feedback_dir/<user_family>/``; рядом пишется
    JSON-манифест ``<имя>.json`` с полями ``request_id``, ``ts``,
    ``user_family``, ``user_state`` и ``origin`` (``"user"``).

    Args:
        file: Загружаемое изображение (JPEG/PNG, максимум из конфига).
        user_family: Правильная семья трактора; валидируется по ``MODEL_CLASSES``.
        request_id: Идентификатор исходного запроса ``/predict`` (опционально).
        user_state: Правильное состояние (clean/dirty), опционально; валидируется
            по ``STATE_CLASSES``.

    Returns:
        Флаг сохранения и путь к сохранённому фото.

    Raises:
        HTTPException: 422 при невалидном расширении, неизвестной семье или
            состоянии, пустом или слишком большом файле.
    """
    _validate_upload(file)

    if user_family not in MODEL_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"Неизвестная семья: {user_family!r}. Доступны: {sorted(MODEL_CLASSES)}.",
        )
    if user_state is not None and user_state not in STATE_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"Неизвестное состояние: {user_state!r}. Доступны: {sorted(STATE_CLASSES)}.",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Пустой файл.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Файл превышает лимит {MAX_FILE_SIZE} байт.",
        )

    target_dir = FEEDBACK_DIR / user_family
    target_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix.lower()
    stem = request_id or uuid4().hex
    photo_path = target_dir / f"{stem}{suffix}"
    photo_path.write_bytes(contents)

    manifest = {
        "request_id": request_id,
        "ts": datetime.now().isoformat(),
        "user_family": user_family,
        "user_state": user_state,
        "origin": "user",
    }
    photo_path.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return FeedbackResponse(saved=True, path=str(photo_path))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
