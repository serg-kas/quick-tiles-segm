#!/usr/bin/env bash
set -e

# Цвета для красивого вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Проверка прав root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Запустите скрипт с sudo:${NC}"
    echo "  sudo bash setup_docker.sh"
    exit 1
fi

# Шаг 1: Проверка драйверов NVIDIA
echo -e "${CYAN}[1/5]${NC} ${GREEN}Проверка драйверов NVIDIA...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ОШИБКА: nvidia-smi не найдена.${NC}"
    echo -e "${YELLOW}Драйверы NVIDIA не установлены или не настроены.${NC}"
    echo "Установите драйверы вручную:"
    echo "  В Ubuntu: sudo apt install nvidia-driver-580 (или последнюю версию)"
    echo "  В WSL2: драйвер ставится в Windows, после чего nvidia-smi заработает в WSL."
    echo "После установки перезапустите этот скрипт."
    exit 1
fi
echo -e "  Видеокарта: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo -e "  Версия драйвера: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# Шаг 2: Docker и Docker Compose
echo -e "${CYAN}[2/5]${NC} ${GREEN}Проверка и установка Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo "  Docker не найден. Устанавливаю..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    echo -e "  ${GREEN}Docker установлен.${NC}"
else
    echo -e "  Docker уже установлен (версия $(docker --version | awk '{print $3}' | tr -d ','))."
fi

# Добавляем пользователя в группу docker (если скрипт запущен через sudo)
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    if ! groups "$SUDO_USER" | grep -q '\bdocker\b'; then
        usermod -aG docker "$SUDO_USER"
        echo -e "  ${YELLOW}Пользователь $SUDO_USER добавлен в группу docker.${NC}"
        echo -e "  ${YELLOW}Для работы без sudo потребуется выйти из сессии и зайти снова (или выполнить 'newgrp docker').${NC}"
    fi
fi

echo -e "  Проверка Docker Compose..."
if ! docker compose version &> /dev/null; then
    echo "  Docker Compose V2 не найден. Устанавливаю как плагин..."
    DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
    mkdir -p $DOCKER_CONFIG/cli-plugins
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f4)
    curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
         -o $DOCKER_CONFIG/cli-plugins/docker-compose
    chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
    echo -e "  ${GREEN}Docker Compose V2 установлен (версия $COMPOSE_VERSION).${NC}"
else
    echo -e "  Docker Compose V2 уже доступен."
fi

# Шаг 3: NVIDIA Container Toolkit
echo -e "${CYAN}[3/5]${NC} ${GREEN}Установка NVIDIA Container Toolkit...${NC}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo -e "${RED}Не удалось определить дистрибутив. Поддерживаются только Ubuntu/Debian.${NC}"
    exit 1
fi

if [ "$OS" != "ubuntu" ] && [ "$OS" != "debian" ]; then
    echo -e "${RED}Скрипт поддерживает только Ubuntu/Debian. Для вашего дистрибутива установите toolkit вручную:${NC}"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    exit 1
fi

# Добавляем репозиторий NVIDIA Container Toolkit (один раз)
if [ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$OS$VER/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
fi

if dpkg -l | grep -q nvidia-container-toolkit; then
    echo -e "  NVIDIA Container Toolkit уже установлен."
else
    apt-get install -y nvidia-container-toolkit
    echo -e "  ${GREEN}Установлен NVIDIA Container Toolkit.${NC}"
fi

# Шаг 4: Настройка Docker-рантайма
echo -e "${CYAN}[4/5]${NC} ${GREEN}Конфигурация Docker для GPU...${NC}"
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
echo -e "  ${GREEN}Конфигурация обновлена, Docker перезапущен.${NC}"

# Шаг 5: Тестовый запуск с GPU
echo -e "${CYAN}[5/5]${NC} ${GREEN}Проверка работы GPU в Docker...${NC}"
echo -e "  Запуск тестового контейнера (nvidia-smi)..."

# Выбираем образ с CUDA 12.8.0 (подходит для драйверов >=525.60.13). Если не работает, пробуем 11.8.
TEST_IMAGE="nvidia/cuda:12.8.0-base-ubuntu22.04"
echo -e "  Пробуем образ $TEST_IMAGE..."

if docker run --rm --gpus all "$TEST_IMAGE" nvidia-smi &> /dev/null; then
    echo -e "  ${GREEN}GPU успешно доступны в контейнере!${NC}"
else
    echo -e "  ${YELLOW}Не удалось запустить с $TEST_IMAGE. Пробую nvidia/cuda:11.8.0-base-ubuntu22.04...${NC}"
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "  ${GREEN}GPU работают (c CUDA 11.8). Всё в порядке.${NC}"
    else
        echo -e "  ${RED}Не удалось запустить nvidia-smi в контейнере. Причины:${NC}"
        echo -e "  - Ваш драйвер слишком старый для CUDA 11.8. Обновите драйвер."
        echo -e "  - Пакет nvidia-container-toolkit установился некорректно."
        echo -e "  Проверьте вручную: docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   Окружение готово!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Теперь вы можете запустить ваше приложение:"
echo -e "  ${CYAN}docker compose up --build${NC}"
echo ""
echo -e "Если вы добавлены в группу docker, но не пере-заходили в сессию,"
echo -e "команды docker могут требовать sudo. Выполните:"
echo -e "  ${YELLOW}newgrp docker${NC}   или просто перезайдите в терминал."

