# Актуальный образ CUDA 12.8 на Ubuntu 22.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Установка Python 3.10, pip, системных библиотек (OpenCV, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements.txt ДО установки пакетов (для кэширования)
COPY requirements.txt .

# Устанавливаем PyTorch 2.8.0 + torchvision 0.23.0 с индексом CUDA 12.8
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Устанавливаем остальные зависимости из requirements.txt (с кэшированием)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile -r requirements.txt

# Устанавливаем SAM2: используем pip install из git (без ручного клонирования)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile git+https://github.com/serg-kas/sam21.git

# Копируем исходный код приложения
COPY src/ ./src/

CMD ["python", "src/app.py"]

