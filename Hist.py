import os

import numpy as np
import pandas as pd
from numpy.ma.bench import xl
import matplotlib.pyplot as plt
from FigureBuilder import add_activations_from_file

'''
def get_lanes_number():
    # Получаем количество полос и детекторов из файла с координатами
    if os.path.exists('Coordinates.csv'):
        df = pd.read_csv('Coordinates.csv')
        # Проверяем не пустой ли файл
        if df.empty:
            print("Ошибка. Файл Coordinates.csv с координатами детекторов пустой")
        else:
            # Получаем координаты детекторов из файла
            columns_list = df.columns.values.tolist()
            buf = columns_list[0].split(';')
            buf = buf[len(buf) - 1].split(' ')
            buf = buf[1].split('.')
            return number_of_lanes = int(buf[0])
            return detectors_per_lane = int(buf[1])


'''


def count_detection_lengths(lanes):
    longest_detection_length = 0
    shortest_detection_length = 0
    max_length = 35
    detections_on_lane = [0 for y in range(max_length)]
    for lane in lanes:
        for detector in lane:
            current_detection_length = 0
            for activation in detector.detections:
                if activation == 1:
                    current_detection_length += 1
                else:
                    if current_detection_length != 0 and current_detection_length < max_length:
                        detections_on_lane[current_detection_length] += 1
                    current_detection_length = 0
    return detections_on_lane


''' 
    df = pd.read_csv('FilteredDetections.csv')
    column = df.iloc[:, 0]
    columns = df.columns
    columns = df.columns[0].split(';')
    for column in range(0, len(columns)):
        single_column = df['lane1 det1']
        single_column = single_column.astype(int)
        single_column = list(map(int, single_column))
        # single_int_column = list(map(int, single_string_column))
        current_detection_length = 0
        for value in single_column:
            if value == 1:
                current_detection_length += 1
            else:
                detections_on_lane[current_detection_length][1] += 1

    for i in range(len(df)):
        single_line_values = df.values[i][0].split(';')
    m = 0

'''
lanes = add_activations_from_file(3, 20, 'FilteredDetections.csv')
detections_length_1 = count_detection_lengths(lanes)
lanes = add_activations_from_file(3, 20, 'E:\PyProjects\Virtual_Detectors\exp3_1 hour_step_2\FilteredDetections.csv')
detections_length_2 = count_detection_lengths(lanes)
lanes = add_activations_from_file(3, 20, 'E:\PyProjects\Virtual_Detectors\exp3_1 hour_step_3\FilteredDetections.csv')
detections_length_3 = count_detection_lengths(lanes)
lanes = add_activations_from_file(3, 20, 'E:\PyProjects\Virtual_Detectors\exp3_1 hour_step_4\FilteredDetections.csv')
detections_length_4 = count_detection_lengths(lanes)
lanes = add_activations_from_file(3, 20, 'E:\PyProjects\Virtual_Detectors\exp3_1 hour_step_5\FilteredDetections.csv')
detections_length_5 = count_detection_lengths(lanes)
# detections_length = [x for x in detections_length if x != 0]


# plt.hist(detections_length, len(detections_length), histtype ='bar')

x = []
for i in range(0, len(detections_length_5)):
    x.append(i)
fig, ax = plt.subplots()

ax.set(xlabel="detection lengths", ylabel="Number of detections")
plt.stem(x, detections_length_1, linefmt='red', markerfmt='o', label='Frame step = 0')
#plt.stem(x, detections_length_2, linefmt='blue', markerfmt='s', label='Frame step = 1')
#plt.stem(x, detections_length_3, linefmt='orange', markerfmt='p', label='Frame step = 2')
#plt.stem(x, detections_length_4, linefmt='purple', markerfmt='P', label='Frame step = 3')
#plt.stem(x, detections_length_5, linefmt='green', markerfmt='D', label='Frame step = 4')

legend = ax.legend()
plt.show()
