FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Установка Python 3 и pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Ссылка python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY src/ ./src/

CMD ["python", "src/app.py"]
