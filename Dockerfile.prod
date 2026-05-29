# Актуальный образ CUDA 12.8 на Ubuntu 22.04
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04


# Установка Python 3.10, pip, системных библиотек (OpenCV, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


# Копируем requirements.txt ДО установки пакетов (для кэширования)
COPY requirements-prod.txt .


# Устанавливаем PyTorch 2.8.0 + torchvision 0.23.0 с индексом CUDA 12.8
# и остальные зависимости
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile numpy==1.26.4 && \
    pip install --no-cache-dir --no-compile \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir --no-compile -r requirements-prod.txt


# Клонируем SAM2 в /app/sam2 (на один уровень с src) и устанавливаем    
#RUN git clone --depth 1 https://github.com/serg-kas/sam21.git /app/sam21 && \
#    pip install --no-cache-dir -e /app/sam21 && \
#    rm -rf /app/sam21/.git
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential python3.10-dev \
    && git clone --depth 1 https://github.com/serg-kas/sam21.git /app/sam21 \
    && pip install --no-cache-dir --no-build-isolation -e /app/sam21 \
    && rm -rf /app/sam21/.git \
    # удаляем build-зависимости и git, чистим мусор
    && apt-get purge -y git build-essential python3.10-dev \
    && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/* /tmp/* /var/tmp/*
        

# Копируем исходный код приложения
COPY src/ ./src/

CMD ["python", "src/app.py"]

