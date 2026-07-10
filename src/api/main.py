"""FastAPI application for Tractor Vision."""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, ModelInfo, PredictionResponse
from src.data.transforms import get_val_transforms
from src.models.classifier import CLASS_NAMES, TractorClassifier
from src.models.multi_task import MultiTaskTractorClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
WEIGHTS_DIR = Path("weights")

single_task_model: Optional[TractorClassifier] = None
multi_task_model: Optional[MultiTaskTractorClassifier] = None
transform = get_val_transforms(IMAGE_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей при старте."""
    global single_task_model, multi_task_model
    logger.info("Loading models...")

    single_task_path = WEIGHTS_DIR / "final_model.ckpt"
    if single_task_path.exists():
        try:
            single_task_model = TractorClassifier(num_classes=len(CLASS_NAMES))
            checkpoint = torch.load(
                single_task_path, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            single_task_model.load_state_dict(state_dict)
            single_task_model.eval()
            logger.info("Loaded single-task model")
        except Exception as e:
            logger.error("Failed to load single-task model: %s", e)

    multi_task_path = WEIGHTS_DIR / "multi_task_final.ckpt"
    if multi_task_path.exists():
        try:
            multi_task_model = MultiTaskTractorClassifier()
            checkpoint = torch.load(
                multi_task_path, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            multi_task_model.load_state_dict(state_dict)
            multi_task_model.eval()
            logger.info("Loaded multi-task model")
        except Exception as e:
            logger.error("Failed to load multi-task model: %s", e)

    yield


app = FastAPI(
    title="Tractor Vision API",
    description="API для классификации тракторов",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Проверка здоровья сервиса."""
    return HealthResponse(
        models_loaded=(single_task_model is not None or multi_task_model is not None)
    )


@app.get("/models")
async def list_models() -> dict:
    """Список доступных моделей."""
    models = []
    if single_task_model is not None:
        models.append(
            ModelInfo(
                name="Single-Task Classifier",
                num_classes=len(CLASS_NAMES),
                accuracy=0.9149,
                weights_path=str(WEIGHTS_DIR / "final_model.ckpt"),
            )
        )
    if multi_task_model is not None:
        models.append(
            ModelInfo(
                name="Multi-Task Classifier",
                num_classes=len(CLASS_NAMES),
                accuracy=0.7917,
                weights_path=str(WEIGHTS_DIR / "multi_task_final.ckpt"),
            )
        )
    return {"models": models, "count": len(models)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Классификация трактора по изображению."""
    start_time = time.time()

    if file.filename is None:
        raise HTTPException(status_code=422, detail="File name is required")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}",
        )

    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=422,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB",
            )
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error reading file: %s", e)
        raise HTTPException(status_code=422, detail=f"Invalid image file: {e}")

    try:
        input_tensor = transform(image=image)["image"].unsqueeze(0)
    except Exception as e:
        logger.error("Error preprocessing image: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    try:
        with torch.no_grad():
            if multi_task_model is not None:
                model_logits, state_logits = multi_task_model(input_tensor)
                model_probs = torch.softmax(model_logits, dim=1)
                state_probs = torch.softmax(state_logits, dim=1)

                model_idx = torch.argmax(model_probs, dim=1).item()
                state_idx = torch.argmax(state_probs, dim=1).item()

                model_class = CLASS_NAMES[model_idx]
                confidence = float(model_probs[0, model_idx].item())
                state = "clean" if state_idx == 0 else "dirty"

            elif single_task_model is not None:
                logits = single_task_model(input_tensor)
                probs = torch.softmax(logits, dim=1)

                model_idx = torch.argmax(probs, dim=1).item()
                model_class = CLASS_NAMES[model_idx]
                confidence = float(probs[0, model_idx].item())
                state = None

            else:
                raise HTTPException(status_code=500, detail="No models loaded")

        processing_time = time.time() - start_time

        logger.info(
            "Prediction: %s (confidence=%.3f, state=%s, time=%.3fs)",
            model_class,
            confidence,
            state,
            processing_time,
        )

        return PredictionResponse(
            model_class=model_class,
            confidence=confidence,
            state=state,
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
