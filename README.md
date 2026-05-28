# Quick tiling segmentation

## Установка и запуск приложения

Возможные варианты запуска приложения:

1. Запуск на python c самостоятельным предварительным развертыванием окружения (режим разработки).
2. В самостоятельно собираемом образе docker (требуется выполнить п.1).
3. В предварительно собранном образе docker (образ скачивается с нашего ресурса)

### Развертывание и запуск на python

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

```bash
  pip install -r requirements.txt

  pip install -e sam21/.
```

7. Первый запуск приложения (запускать из папка проекта)

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

Режим работы можно писать сокращенно, например baseline вместо baseline_workflow 

Возможно подавать в обработку изображения из других папок, передавая путь в параметрах.

```bash
python src/app.py baseline source2 outfolder2
```

Брать файлы для обработки в source2, результаты помещать в outfolder2

11. Прочие параметры приложения передаются через переменные окружения

```
    Скопируйте файл cfg-example.env в файл cfg.env в папке приложения.  
    Отредактируйте в нем необходимые параметры, например установите  
    APP_SAM2_FORCE_CUDA="False" для запрета использовать GPU
```

12. Обновление приложения из репозитория

```bash
git pull
```

Набирать команду из папки проекта.

### Самостоятельная сборка и запуск образа docker

В системе должны быть установлены драйвера NVIDIA, 
NVIDIA Container Toolkit, Docker и Docker Compose.  

Скрипт setup_docker.sh проверит установку необходимых компонентов и попытается установить отсутствующие.

Работает с ubuntu/debian, в том числе в Windows WSL2
Запускать с sudo:
```bash
sudo ./setup_docker.sh
```

В случае успешной проверки и/или установки компонентов будет скачан и запущен 
образ для проверки доступности GPU из программы в docker.

Проверку доступности GPU можно запустить вручную командой:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Команда сборки образа приложения:
```bash
docker build -t segm-tiles-image .
```

Команда запуска приложения в docker.
Предварительно надо создать папки source_files и out_files если их еще нет.
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

### Запуск предварительно собранного образа docker

Для удобства запуск приложения можно проводить через docker compose.

Команда запуска приложения в docker compose (запустится режим test):
```bash
docker compose up
```

Команда запуска приложения в docker compose в другом режиме:
```bash
docker compose run app python src/app.py baseline
```

Чтобы установить другой режим работы по умолчанию можно отредактировать соответствующее место в конфигурационном файле docker-compose.yaml (например заменить test на tiling)
```
...
command: python src/app.py test # заменить на tiling
...
```

### Прочие команды, которые могут быть полезны

Чтобы присвоить образу тэг и использовать указанный Dockerfile вместо файла по умолчанию:
```bash
docker build -t segm-tiles-image:dev -f Dockerfile.dev .
```

Команда интерактивного запуска образа:
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

Запуск через docker compose с очисткой следов предыдущих запусков
```bash
docker compose up --remove-orphans
```
или
```bash
docker compose run --remove-orphans app python src/app.py baseline
```

Очистить docker, удалив всё неиспользуемые образы
```bash
docker system prune -a
```
