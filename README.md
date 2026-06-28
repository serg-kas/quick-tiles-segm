# Quick tiling segmentation

## Установка и запуск приложения

Возможные варианты запуска приложения:

1. Запуск на python c самостоятельным предварительным развертыванием окружения (режим разработки).
2. В самостоятельно собираемом образе docker (требуется выполнить п.1, потом собрать образ).
3. В готовом образе docker (образ скачивается с Docker Hub, п.п.1 и 2 выполнять не нужно).

### Развертывание и запуск на python.

1. Клонируем репозиторий и переходим в папку проекта

```bash
git clone https://github.com/serg-kas/quick-tiles-segm.git

cd quick-tiles-segm
```

2. Внутри папки проекта клонируем репозиторий модели 

```bash
git clone https://github.com/serg-kas/sam21.git
```

3. Идем в папку models и скачиваем веса (в папке есть свой README)

```bash
сd models  
./download_ckpts.sh или ./download_yandex.sh
```
В папке models должен появиться файл sam2.1_hiera_large.pt

4. Выходим на уровень папки проекта

```bash
cd ..
```

5. Создаем питоновское окружение и активируем его (пример на conda)

```bash
conda create -n py310 -c conda-forge python=3.10 pip

conda activate py310
```

6. Устанавливаем зависимости

Для разработки надо устанавливать зависимости из requirements-dev.txt
```bash
  pip install -r requirements-dev.txt

  pip install -e sam21/.
```

7. Первый запуск приложения (запускать из папки проекта)

```bash
python src/app.py help
```

Выводится список доступных режимов работы.

Будут созданы папки source_files и out_files для исходных и готовых изображений.

8. Тестовый запуск приложения

```bash
python src/app.py test
```

Будет сделан пробный запуск модели на GPU при его наличии.

Выводятся результаты запуска и характеристики доступного GPU.  

9. Рабочий запуск приложения

Поместить изображение (одно или несколько) в source_files.

По окончании обработки результаты будут помещены в папку out_files  

```bash
python src/app.py baseline_workflow
```

Режим работы можно писать сокращенно, например baseline вместо baseline_workflow, tiling вместо workflow_tiling.  

Возможно подавать в обработку изображения из других папок, передавая путь в параметрах.  
Например, файлы для обработки берутся в папке source2, результаты помещаются в папку outfolder2.  

```bash
python src/app.py baseline source2 outfolder2
```

11. Прочие параметры приложения передаются через переменные окружения

```
    Скопируйте файл cfg-example.env в файл cfg.env в папке приложения.  
    Отредактируйте в нем необходимые параметры, например установите  
    APP_SAM2_FORCE_CUDA="" для запрета использовать GPU
```

12. Обновление приложения из репозитория (набирать команду из папки проекта).

```bash
git pull
```

### Самостоятельная сборка и запуск образа docker.

В системе должны быть установлены драйвера NVIDIA, 
NVIDIA Container Toolkit, Docker и Docker Compose.  

Скрипт setup_docker.sh проверит установку необходимых компонентов и попытается установить отсутствующие.

Работает с ubuntu/debian, в том числе в Windows WSL2
Запускать с sudo:
```bash
sudo ./setup_docker.sh
```
Примечание:
```
Гарантировать, что скрипт setup_docker.sh успешно установит всё необходимое для сборки docker образа  
на конкретном компьютере нельзя, поэтому в случае необходимости нужно установить необходимые компоненты по отдельности.  
```

В случае успешной проверки и/или установки компонентов будет скачан и запущен 
образ для проверки доступности GPU из программы в docker.

Проверку доступности GPU можно запустить командой:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Команда сборки образа приложения:
```bash
docker build -t segm-tiles-image .
```

Необходимо создать папки для исходных и готовых изображений source_files и out_files,
если они не создавались ранее.  В папке models должны быть скачаны веса модели.
```bash
mkdir -p source_files out_files
```  

Команда запуска приложения в docker из локально собранного образа.
```bash
docker run -it --rm \
  --gpus all \
  --env-file cfg.env \
  -v "$(pwd)/source_files:/app/source_files" \
  -v "$(pwd)/out_files:/app/out_files" \
  -v "$(pwd)/models:/app/models" \
  segm-tiles-image \
  python src/app.py test
```

Для удобства запуск приложения можно проводить через docker compose.

Команда запуска приложения в docker compose (запустится режим test):
```bash
docker compose -f docker-compose-dev.yaml up
```

Команда запуска приложения в docker compose в другом режиме (baseline):
```bash
docker compose -f docker-compose-dev.yaml run app python src/app.py baseline
```


### Запуск предварительно собранного образа из Docker Hub. 

Скачать в систему готовый образ из Docker Hub:
```bash
docker pull sergkas/segm-tiles-image:latest
```

Посмотреть образы docker в системе:
```bash
docker images
```
Среди образов в системе вы увидите:
```
sergkas/segm-tiles-image:latest
```

Необходимо создать папки для исходных и готовых изображений source_files и out_files,
если они не создавались ранее.  В папке models должны быть скачаны веса модели.
```bash
mkdir -p source_files out_files
```  

Команда запуска в docker аналогична запуску локально собранного образа:
```bash
docker run -it --rm \
  --gpus all \
  --env-file cfg.env \
  -v "$(pwd)/source_files:/app/source_files" \
  -v "$(pwd)/out_files:/app/out_files" \
  -v "$(pwd)/models:/app/models" \
  sergkas/segm-tiles-image:latest \
  python src/app.py test
```
Внимание: используется имя образа с Docker Hub (sergkas/segm-tiles-image:latest).  

Запуск через docker compose аналогичен запуску локально собранного образа:
```bash
docker compose run app python src/app.py baseline
```


### Как запустить программу из Docker Hub при наличии настроенной системы (кратко).
1. Cоздать папку для приложения и перейти в нее:
```bash
mkdir -p quick-tiles-segm
cd quick-tiles-segm
```
2. Создать папку models и положить в нее файл весов модели, который можно взять по ссылке:
```
https://disk.yandex.ru/d/cMsiauLhyvhsgg
```
3. Создать папки для исходных и готовых изображений source_files и out_files.
```bash
mkdir -p source_files out_files
```  
4. Создать файл docker-compose.yaml с содержимым:
```
services:
  app:
    image: segm-tiles-image
    env_file: cfg.env
    volumes:
      - ./source_files:/app/source_files
      - ./out_files:/app/out_files
      - ./models:/app/models
    command: python src/app.py test
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```
5. Создать файл с настройками cfg.env: 
```
# Переменные окружения должны иметь имя APP_ + имя переменной в верхнем регистре
# Параметр False надо передавать пустой строкой: ""

# Выводить дополнительную информацию
APP_VERBOSE="True"

# Использовать GPU
APP_SAM2_FORCE_CUDA="True"
```
6. Запустить программу в тестовом режиме командой:
```bash
docker compose up
```
При первом запуске образ будет автоматически загружен с Docker Hub.

6. В папку source_files помещаем изображение для обработки и запускаем программу в рабочем режиме:  
```bash
docker compose run app python src/app.py baseline
```

Примечание:  
Архив quick-tiles-segm-docker-run.zip содержит минимальный набор файлов для запуска приложения из готового образа

```bash
unzip quick-tiles-segm-docker-run.zip
```


### Существует образ, работающий только на CPU. 

Команда запуска приложения на CPU в docker compose в режиме baseline.
```bash
docker compose -f docker-compose-cpu.yaml run app python src/app.py baseline
```
Примечание:
```
Работа приложения без использования NVIDIA будет медненней в несколько раз. 
Использование приложения без GPU не рекомендуется для практического применения, но с его помощью можно
проверить работоспобность остальных компонентов системы.
```


### Прочие команды, которые могут быть полезны (не исчерпывающий список).

Чтобы присвоить образу тэг и использовать указанный Dockerfile вместо файла по умолчанию:
```bash
docker build -f Dockerfile.cpu -t segm-tiles-image:cpu .
```
В примере сборка образа для работы на CPU.

Команда интерактивного запуска образа (запуск в командной строке):
```bash
docker run -it --rm \
  --gpus all \
  --env-file cfg.env \
  -v "$(pwd)/source_files:/app/source_files" \
  -v "$(pwd)/out_files:/app/out_files" \
  -v "$(pwd)/models:/app/models" \
  --entrypoint /bin/bash \
  segm-tiles-image
```

Запуск через docker compose с очисткой следов предыдущих запусков:
```bash
docker compose up --remove-orphans
```
или
```bash
docker compose run --remove-orphans app python src/app.py baseline
```

Очистить кэш сборки (не затрагивает образы)
```bash
docker builder prune -af
```

Очистить docker, удалив все образы, кэш и т.д.
```bash
docker system prune -a
```

Утилита для анализа структуры образа
```bash
docker pull wagoodman/dive
docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock wagoodman/dive segm-tiles-image:latest
```
