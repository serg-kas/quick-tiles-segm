# Актуальный образ CUDA 12.4 на Ubuntu 22.04
#FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
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

# Устанавливаем PyTorch с CUDA 12.1 (отдельно, чтобы не качать из requirements)
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install --no-cache-dir --no-compile \
#    torch==2.8.0 torchvision==0.23.0 \
#    --index-url https://download.pytorch.org/whl/cu121
# Устанавливаем PyTorch 2.8.0 + torchvision 0.23.0 с индексом CUDA 12.8
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128


# Устанавливаем остальные зависимости из requirements.txt (с кэшированием)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile -r requirements.txt


# Клонируем модель SAM2
#RUN pip install --no-cache-dir git+https://github.com/serg-kas/sam21.git
#RUN git clone https://github.com/serg-kas/sam21.git /app/sam21

# Клонируем SAM2 в /app/sam2 (на один уровень с src) и устанавливаем
#RUN git clone https://github.com/serg-kas/sam21.git /app/sam21 && \
#    cd /app/sam21 && \
#    # Если есть setup.py — устанавливаем в editable режиме
#    (test -f setup.py && pip install --no-cache-dir -e . || true) && \
#    rm -rf /app/sam21/.git

# Добавляем /app в PYTHONPATH, чтобы import sam2 работал (если нет setup.py)
#ENV PYTHONPATH="/app:${PYTHONPATH}"


# Устанавливаем SAM2: используем pip install из git (без ручного клонирования)
# Это самый чистый способ: не оставляет .git, исходники не дублируются
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-compile git+https://github.com/serg-kas/sam21.git


# Официальный PyPI
 #RUN pip install --no-cache-dir -r requirements.txt

# SAM21
 #RUN pip install --no-cache-dir -e sam21/.

# Официальный PyPI + таймауты + оптимизация
#RUN pip install --no-cache-dir \
#    --no-compile \
#    --default-timeout=100 \
#    --retries=5 \
#    -r requirements.txt \
#    && find /usr/local/lib/python*/dist-packages/ -name "*.pyc" -delete \
#    && rm -rf /root/.cache/pip
 

# Гибридный вариант: быстрое зеркало (Aliyun), при ошибке — официальный PyPI
#RUN pip install --no-cache-dir \
#    --no-compile \
#    --default-timeout=100 \
#    --retries=5 \
#    -i https://mirrors.aliyun.com/pypi/simple/ \
#    --extra-index-url https://pypi.org/simple \
#    --trusted-host mirrors.aliyun.com \
#    -r requirements.txt \
#    && find /usr/local/lib/python*/dist-packages/ -name "*.pyc" -delete \
#    && rm -rf /root/.cache/pip


# Копируем исходный код приложения
COPY src/ ./src/

CMD ["python", "src/app.py"]

