import math

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
from numpy import exp
#from scipy.interpolate import interp1d
#from scipy import interpolate
#from scipy.optimize import curve_fit


#######################################################################################################################
# Подсчет частоты кадров
# Подсчет частоты кадров


def get_frame_rate():
    # Получаем список всех файлов в папке
    video_path = 'Videos/'
    files = os.listdir(video_path)
    videos = []

    # Добавляем в список только файлы с нужным расширением
    for file in files:
        if file.endswith(".mp4"):
            videos.append("Videos/" + file)

    video = cv.VideoCapture(videos[0])
    fps = video.get(cv.CAP_PROP_FPS)
    return fps


frame_rate = math.floor(get_frame_rate())  # Частота кадров


########################################################################################################################
# Класс Детектор
class Detector:

    def __init__(self):
        self.detX = 0
        self.detY = 0
        self.avgColour = []
        self.detections = []


#######################################################################################################################
# Добавляем пустые детекторы на полосы
def add_activations_from_file(number_of_lanes, detectors_per_lane, filename):
    lanes = []  # Список полос
    detectors = []  # Список детекторов на одну полосу
    for i in range(number_of_lanes):
        for j in range(detectors_per_lane):
            detectors.append(Detector())
        lanes.append(detectors.copy())
        detectors.clear()

    # Добавляем детекторам значения активации из файла
    df = pd.read_csv(filename)
    for i in range(len(df)):
        single_line_values = df.values[i][0].split(';')
        m = 0
        for j in range(len(lanes)):
            for k in range(len(lanes[j])):
                lanes[j][k].detections.append(int(single_line_values[m]))
                m += 1
    return lanes


########################################################################################################################
# Обрезаем список активаций, чтобы он был кратен частоте кадров
def cut_data(lanes, time_slice):
    fps = math.floor(get_frame_rate())
    for lane in lanes:
        for detector in lane:
            trim_frames = len(detector.detections) % fps
            while trim_frames > 0:
                detector.detections.pop()
                trim_frames -= 1
    return lanes


########################################################################################################################
# Если продолжительность видео меньше заданного интервала
def cut_time_slice(lanes, time_slice):
    video_length = len(lanes[0][0].detections) / frame_rate
    if time_slice > video_length:
        # Устанавливаем интервал равным продолжительности видео
        time_slice = int(video_length)
    return time_slice


########################################################################################################################
# Подсчет плотности по кадрам
def get_density(lanes):
    density_per_lane = []  # Список значений плотности на полосу
    density_lanes = []  # Список значений плотности на полосы
    detections_counter = 0
    for i in range(len(lanes)):
        for k in range(len(lanes[i][0].detections)):
            for j in range(len(lanes[i])):
                if lanes[i][j].detections[k] == 1:
                    detections_counter += 1.0
            density_per_lane.append(detections_counter / len(lanes[i]))
            detections_counter = 0
        density_lanes.append(density_per_lane.copy())
        density_per_lane.clear()
    return density_lanes


########################################################################################################################
# Перевод плотности в секунды
def density_to_sec(density_lanes):
    density_per_lane_sec = []  # Список значений плотности для полосы в сек.
    density_lanes_sec = []  # Список значений плотности для полос в сек.
    for lane in density_lanes:
        frame_counter = 0
        seconds = 1
        while frame_counter <= len(lane):
            frame_sum = 0
            for i in range(frame_rate * (seconds - 1), frame_rate * seconds):
                if i < len(lane):
                    frame_sum += lane[i]
                frame_counter += 1
            density_per_lane_sec.append(frame_sum / frame_rate)
            seconds += 1
        density_lanes_sec.append(density_per_lane_sec.copy())
        density_per_lane_sec.clear()
    return density_lanes_sec


########################################################################################################################
# Заполняем csv файл значениями плотности для каждой полосы
def fill_density(density_lanes_sec):
    data = dict()
    lane_counter = 1
    for lane in density_lanes_sec:
        new_dict = {"lane" + str(lane_counter): lane}
        data.update(new_dict)
        lane_counter += 1
    df = pd.DataFrame(data)
    df.to_csv(r'Density.csv', sep=';', index=False)


########################################################################################################################
# Построение графиков плотности по секундам
def draw_density(density_lanes_sec, number_of_lanes):
    x = []
    for i in range(0, len(density_lanes_sec[0])):
        x.append(i)

    if number_of_lanes > 1:
        fig, axs = plt.subplots(number_of_lanes)
        i = 0
        for lane in density_lanes_sec:
            j = 0
            axs[i].step(x, lane)
            axs[i].set_title("Lane №" + str(i))
            i += 1
        for ax in axs.flat:
            ax.set(xlabel="Frames", ylabel="Density")
        plt.show()
    else:
        plt.step(x, density_lanes_sec[0])
        plt.show()


########################################################################################################################
# Подсчет уникальных активаций детекторов за time_slice
def get_activations_per_time_slice(lanes, time_slice):
    detections_per_detector = []  # Активации одного детектора за time_slice
    detections_per_lane = []  # Активации детекторов полосы за time_slice
    detections_per_lanes = []  # Активации детекторов всех полос за time_slice
    for lane in lanes:
        for detector in lane:
            for i in range(0, len(detector.detections), frame_rate * time_slice):
                detections_counter = 0
                j = 0
                while (i < len(detector.detections)) and (j < frame_rate * time_slice):
                    if (i + j < len(detector.detections)) and (detector.detections[i + j] == 1):
                        while (i + j < len(detector.detections)) and (j < frame_rate * time_slice) and (
                                detector.detections[i + j]) == 1:
                            j += 1
                        detections_counter += 1
                    else:
                        j += 1
                detections_per_detector.append(detections_counter)
            detections_per_lane.append(detections_per_detector.copy())
            detections_per_detector.clear()
        detections_per_lanes.append(detections_per_lane.copy())
        detections_per_lane.clear()
    return detections_per_lanes


########################################################################################################################
# Заполняем csv файл активациями детекторов за time_slice
def fill_activations_per_time_slice(detections_per_lanes):
    data = dict()
    lane_counter = 1
    for lane in detections_per_lanes:
        detectorNumber = 1
        for detector in lane:
            new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber): detector}
            data.update(new_dict)
            detectorNumber += 1
        lane_counter += 1
    df = pd.DataFrame(data)
    df.to_csv(r'Intensity.csv', sep=';', index=False)


########################################################################################################################
# Подсчет среднего количества активаций на полосу за time_slice
def get_intensity(detections_per_lanes):
    intensity_per_lane = []  # Список значений интенсивности для полосы
    intensity_per_lanes = []  # Список значений интенсивности для полос
    for lane in detections_per_lanes:
        avg_detections = 0
        for detections in lane:
            avg_detections += sum(detections)
            avg_detections /= len(lane)

    detections_counter = 0
    for i in range(len(detections_per_lanes)):
        for j in range(len(detections_per_lanes[i][0])):
            avg_detections = 0
            for k in range(len(detections_per_lanes[i])):
                avg_detections += detections_per_lanes[i][k][j]
            avg_detections /= len(detections_per_lanes[i])
            intensity_per_lane.append(avg_detections)
        intensity_per_lanes.append(intensity_per_lane.copy())
        intensity_per_lane.clear()
    return intensity_per_lanes


########################################################################################################################
# Построение графиков интенсивности по секундам
def draw_intensity(intensity_per_lanes, number_of_lanes):
    x = []
    for i in range(1, len(intensity_per_lanes[0]) + 1):
        x.append(i)

    if number_of_lanes > 1:
        fig, axs = plt.subplots(number_of_lanes)
        i = 0
        for lane in intensity_per_lanes:
            j = 0

            # 1 способ
            t = np.polyfit(x, lane, len(x) - 1)
            f = np.poly1d(t)

            axs[i].plot(x, lane, 'o', x, f(x), '--')
            axs[i].set_title("Lane №" + str(i))
            i += 1
        for ax in axs.flat:
            ax.set(xlabel="Minutes", ylabel="Intensity")
        plt.show()

    else:
        plt.step(x, intensity_per_lanes[0])
        plt.show()


def mapping(x, a, b, c):
    return a * x + b


def mapping1(x, a, b, c):
    return a * x ** 2 + b * x + c


def mapping2(values_x, a, b, c):
    return a * values_x ** 3 + b * values_x + c


def mapping3(values_x, a, b, c):
    return a * values_x ** 3 + b * values_x ** 2 + c


def mapping4(x, a, b, c):
    return a * exp(b * x) + c
