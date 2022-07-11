import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
frame_rate = 30  # Частота кадров
number_of_lanes = 3  # Количество полос
detectors_per_lane = 4  # Количество детекторов на полосу
lanes = []  # Список полос
detectors = []  # Список детекторов на одну полосу
density_per_lane = []  # Список значений плотности на полосу
density_lanes = []  # Список значений плотности на полосы
density_per_lane_sec = []  # Список значений плотности для полосы в сек.
density_lanes_sec = []  # Список значений плотности для полос в сек.

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
            lanes[j][k].detections.append(single_line_values[m])
            m += 1

########################################################################################################################
# Подсчет плотности по кадрам
detections_counter = 0
for i in range(len(lanes)):
    for k in range(len(lanes[i][0].detections)):
        for j in range(len(lanes[i])):
            if lanes[i][j].detections[k] == '1':
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
    ax.set(xlabel="Frames", ylabel="Detections")
plt.show()
