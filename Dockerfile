# syntax=docker/dockerfile:1
#
# Tractor Vision — образ для инференса и тестов.
#
# Обучение идёт локально в conda-окружении `tractor` и в Docker не выполняется,
# поэтому по умолчанию образ CPU-only. Вариант PyTorch задаётся build-arg
# TORCH_VARIANT:
#   * cpu   (по умолчанию) — колёса CPU-only, образ работает на любой машине;
#   * cu121 — колёса CUDA 12.1 (GPU-инференс).
#
# Сборка:
#   docker build -t tractor-vision:cpu .
#   docker build --build-arg TORCH_VARIANT=cu121 -t tractor-vision:gpu .

FROM python:3.11-slim

# Вариант сборки PyTorch: cpu | cu121. Переопределяется через --build-arg.
ARG TORCH_VARIANT=cpu

WORKDIR /app

# Системные зависимости для OpenCV (headless) и компиляции нативных колёс.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем только файлы зависимостей — слой кэшируется отдельно от кода.
COPY requirements-docker.txt .

# Индекс колёс PyTorch зависит от варианта. Для CPU и CUDA у PyTorch разные
# extra-index-url; всё остальное ставится с обычного PyPI.
#   cpu   -> https://download.pytorch.org/whl/cpu
#   cu121 -> https://download.pytorch.org/whl/cu121
RUN TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    && echo "Устанавливаю PyTorch вариант: ${TORCH_VARIANT} (индекс: ${TORCH_INDEX})" \
    && pip install --no-cache-dir -r requirements-docker.txt \
        --extra-index-url "${TORCH_INDEX}"

# Копируем код проекта.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY config.yaml ./config.yaml

# Директории для артефактов (веса, данные, вывод оценки).
RUN mkdir -p weights data output

# По умолчанию образ поднимает API-инференс. В compose команда при
# необходимости переопределяется (например, на прогон тестов).
CMD ["python", "-m", "src.api.main"]
