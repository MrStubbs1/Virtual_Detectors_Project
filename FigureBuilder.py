import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
frame_rate = 30  # Частота кадров
number_of_lanes = 3  # Количество полос
detectors_per_lane = 4  # Количество детекторов на полосу
time_slice = 10  # Интервал времени в сек. для интенсивности
lanes = []  # Список полос
detectors = []  # Список детекторов на одну полосу
density_per_lane = []  # Список значений плотности на полосу
density_lanes = []  # Список значений плотности на полосы
density_per_lane_sec = []  # Список значений плотности для полосы в сек.
density_lanes_sec = []  # Список значений плотности для полос в сек.
detections_per_detector = []  # Активации одного детектора за time_slice
detections_per_lane = []  # Активации детекторов полосы за time_slice
detections_per_lanes = []  # Активации детекторов всех полос за time_slice
intensity_per_lane = []  # Список значений интенсивности для полосы
intensity_per_lanes = []  # Список значений интенсивности для полос


########################################################################################################################
# Перевод плотности в секунды
def density_to_sec(density_lanes):
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


########################################################################################################################
# Класс Детектор
class Detector:

    def __init__(self):
        self.detX = 0
        self.detY = 0
        self.avgColour = []
        self.detections = []


########################################################################################################################
# Добавляем пустые детекторы на полосы
for i in range(number_of_lanes):
    for j in range(detectors_per_lane):
        detectors.append(Detector())
    lanes.append(detectors.copy())
    detectors.clear()

# Добавляем детекторам значения активации из файла
df = pd.read_csv('FilteredDetections.csv')
for i in range(len(df)):
    single_line_values = df.values[i][0].split(';')
    m = 0
    for j in range(len(lanes)):
        for k in range(len(lanes[j])):
            lanes[j][k].detections.append(int(single_line_values[m]))
            m += 1

########################################################################################################################
# Обрезаем список активаций, чтобы он был кратен частоте кадров
for lane in lanes:
    for detector in lane:
        trim_frames = len(detector.detections) % frame_rate
        while trim_frames > 0:
            detector.detections.pop()
            trim_frames -= 1

########################################################################################################################
# Если продолжительность видео меньше заданного интервала
video_length = len(lanes[0][0].detections) / frame_rate
if time_slice > video_length:
    # Устанавливаем интервал равным продолжительности видео
    time_slice = int(video_length)
########################################################################################################################
# Подсчет плотности по кадрам
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

########################################################################################################################
# Построение графиков плотности по секундам
density_to_sec(density_lanes)

x = []
for i in range(0, len(density_lanes_sec[0])):
    x.append(i)

fig, axs = plt.subplots(3)
i = 0
for lane in density_lanes_sec:
    j = 0
    axs[i].step(x, lane)
    axs[i].set_title("Lane №" + str(i))
    i += 1
for ax in axs.flat:
    ax.set(xlabel="Frames", ylabel="Density")
plt.show()

########################################################################################################################
# Подсчет уникальных активаций детекторов за time_slice
for lane in lanes:
    for detector in lane:
        for i in range(0, len(detector.detections), frame_rate * time_slice):
            detections_counter = 0
            j = 0
            while (i < len(detector.detections)) and (j < frame_rate * time_slice):
                if (i + j < len(detector.detections)) and (detector.detections[i + j] == 1):
                    while (i + j < len(detector.detections)) and (j < frame_rate * time_slice) and (detector.detections[i + j]) == 1:
                        j += 1
                    detections_counter += 1
                else:
                    j += 1
            detections_per_detector.append(detections_counter)
        detections_per_lane.append(detections_per_detector.copy())
        detections_per_detector.clear()
    detections_per_lanes.append(detections_per_lane.copy())
    detections_per_lane.clear()

########################################################################################################################
# Подсчет среднего количества активаций на полосу за time_slice
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

########################################################################################################################
# Построение графиков интенсивности по секундам

x = []
for i in range(1, len(intensity_per_lanes[0])+1):
    x.append(i)

fig, axs = plt.subplots(3)
i = 0
for lane in intensity_per_lanes:
    j = 0
    axs[i].step(x, lane)
    axs[i].set_title("Lane №" + str(i))
    i += 1
for ax in axs.flat:
    ax.set(xlabel="Frames", ylabel="Intensity")
plt.show()