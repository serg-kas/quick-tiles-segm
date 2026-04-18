"""
Функции, связанные с вызовом модели SAM2 и обработкой результатов.
Используются оригинальные модели в формате .pt
"""
# import numpy as np
import cv2 as cv
import time
#
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# #########################################################
# Функции для работы с моделью SAM2
# #########################################################
def get_model_sam2(config_file,
                   model_file,
                   force_cuda=False,
                   verbose=False):
    """
    Загрузка модели из указанного файла

    :param config_file: путь к файлу конфигурации модели для загрузки
    :param model_file: путь к файлу чекпоинта модели для загрузки
    :param force_cuda: использовать CUDA
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print("  Загружаем файл конфигурации модели: {}".format(config_file))
        print("  Загружаем файл чекпоинта модели: {}".format(model_file))
    time_0 = time.perf_counter()

    #
    DEVICE = torch.device('cpu')
    if force_cuda:
        if torch.cuda.is_available():
            if verbose:
                print("  Found CUDA, trying to use it")
            #
            DEVICE = torch.device('cuda')
            #
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            #
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            if verbose:
                print("  No CUDA available, using CPU")
    else:
        if verbose:
            print("  Ignoring CUDA, using CPU")
    #
    model = build_sam2(config_file, model_file, device=DEVICE, apply_postprocessing=False)
    #
    time_1 = time.perf_counter()
    if verbose:
        print("  Время загрузки модели, с: {:.2f}".format(time_1 - time_0))
    return model


def get_mask_generator(model, verbose=False):
    """
    Инициализация генератора масок

    :param model: модель
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print("  Инициализируем генератор масок")
    time_0 = time.perf_counter()
    #
    mask_generator = SAM2AutomaticMaskGenerator(model)
    #
    time_1 = time.perf_counter()
    if verbose:
        print("  Время подготовки генератора масок, с: {:.2f}".format(time_1 - time_0))
    return mask_generator


def get_predictor(model, verbose=False):
    """
    Инициализация предиктора

    :param model: модель
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print("Инициализируем предиктор")
    time_0 = time.perf_counter()
    #
    predictor = SAM2ImagePredictor(model)
    #
    time_1 = time.perf_counter()
    if verbose:
        print("  Время подготовки предиктора, с: {:.2f}".format(time_1 - time_0))
    return predictor
