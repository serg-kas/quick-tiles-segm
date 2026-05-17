FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04


# Системные зависимости для OpenCV и Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
#    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ссылка python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Копируем только requirements.txt (кэширование слоёв)
COPY requirements.txt .


# Официальный PyPI
# RUN pip install --no-cache-dir -r requirements.txt

# Официальный PyPI + таймауты + оптимизация
 RUN pip install --no-cache-dir \
    --no-compile \
    --default-timeout=100 \
    --retries=5 \
    -r requirements.txt \
    && find /usr/local/lib/python*/dist-packages/ -name "*.pyc" -delete \
    && rm -rf /root/.cache/pip


# Устанавливаем Python-пакеты с оптимизациями
# RUN pip install --no-cache-dir \
#    --no-compile \
#    -i https://pypi.tuna.tsinghua.edu.cn/simple \
#    --trusted-host pypi.tuna.tsinghua.edu.cn \
#    -r requirements.txt \
#    && find /usr/local/lib/python*/dist-packages/ -name "*.pyc" -delete \
#    && rm -rf /root/.cache/pip    

# Код приложения
COPY src/ ./src/

CMD ["python", "src/app.py", "test"]
