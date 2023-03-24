import cv2 as cv
import numpy as np


########################################################################################################################
# Фильтр для дискретизации активаций детекторов
def detectors_discretization_filter(detectors, frames_unite):
    # Проходим по всем детекторам полосы
    for i in range(0, len(detectors)):
        for j in range(0, len(detectors[i].detections)-1):
            detector_activations_count = 0  # Счетчик активаций детектора
            # Смотрим на 1...frames_unite кадров вперед от текущего
            for k in range(1, frames_unite):
                # Если детектор активирован на текущем и 1...frames_unite кадре
                if (detectors[i].detections[j] == 1 and ((j + k) < (len(detectors[i].detections) - k - 1)) and (
                        detectors[i].detections[j + k] == 1)):
                    # Увеличиваем счетчик срабатываний детектора
                    detector_activations_count += 1
                # Вручную активируем детектор на detector_activations_count кадров от текущего
                for l in range(j, j + detector_activations_count):
                    detectors[i].detections[l] = 1

    #  Еще раз проходим по всем детекторам полосы
    for i in range(0, len(detectors)):
        # Смотрим значения активации детекторов для предыдущего и следующего кадров
        for j in range(0, len(detectors[i].detections) - 2):
            neighbours_sum = detectors[i].detections[j - 1] + detectors[i].detections[j + 1]
            # Если активации отсутствуют, то деактивируем детектор на текущем кадре
            if detectors[i].detections[j] == 1 and neighbours_sum == 0:
                detectors[i].detections[j] = 0


########################################################################################################################
# Дискретизация активаций детекторов
def detectors_discretization(detectors, activation_avg_colour_delta):
    # Задаем начальное значение активации всех детекторов на всех кадрах как 0
    for detector in detectors:
        detector.detections = [0] * len(detector.avgColour)

    # Проходимся по всем кадрам детектора
    for i in range(0, len(detectors)):
        for j in range(0, len(detectors[i].avgColour)-1):
            # Считаем текущую разницу среднего цвета между текущим и предыдущим кадрами
            current_avg_colour_delta = abs(
                (detectors[i].avgColour[j] - detectors[i].avgColour[j + 1]) / detectors[i].avgColour[j]) * 100
            # Если разница больше заданной, то активируем детектор
            if current_avg_colour_delta > activation_avg_colour_delta:
                detectors[i].detections[j + 1] = 1
    return detectors


########################################################################################################################
# Получение среднего значения цвета
def get_avg_colour_sum(gray, detectors, width, height):
    # для всех детекторов
    for detector in detectors:
        # Вырезаем область детектора
        detector_zone = gray[int((detector.detY - (height / 2))):int((detector.detY + (height / 2))),
                        int((detector.detX - (width / 2))):int((detector.detX + (width / 2)))]
        avg_color_per_row = np.average(detector_zone, axis=0)  # считаем средние цвета для пикселей по горизонтали
        avg_color = np.average(avg_color_per_row, axis=0)  # считаем средний цвет для средних цветов по горизонтали
        # Добавляем значение среднего цвета для детектора
        detector.add_avg_colour_sum(avg_color)

        # Отображаем окно для проверки детектора
        show = False
        if show:
            cv.namedWindow('detector', cv.WINDOW_NORMAL)
            cv.imshow('detector', detector_zone)

########################################################################################################################
