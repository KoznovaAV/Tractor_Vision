FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для OpenCV и компиляции
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Копируем код проекта
COPY src/ ./src/
COPY tests/ ./tests/

# Создаём директории
RUN mkdir -p weights data output

# Команда по умолчанию
CMD ["python", "-m", "src.training.train"]
