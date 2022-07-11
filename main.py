import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

########################################################################################################################
# Класс Детектор

class Detector:

    def __init__(self, x, y):
        self.detX = x  # Координата по x
        self.detY = y  # Координата по y
        self.avgColour = []  # Список средних значений цвета для детектора
        self.detections = []  # Список срабатываний детектора 0/1

    # Добавление среднего значения цвета для детектора
    def add_avg_colour_sum(self, value):
        self.avgColour.append(value)


########################################################################################################################
drawing = False
mouseX, mouseY = -1, -1  # Переменные для хранения координат курсора мыши
size = 25  # Размер детектора в пикселях

number_of_lanes = 3  # количество полос
detectors_per_lane = 4  # количество детекторов на полосу

lanes = []  # Список полос
detectors = []  # Список детекторов


########################################################################################################################
# Фильтр для дискретизации активаций детекторов
def detectors_discretization_filter(detectors, frameCounter):
    frames_unite = 10  # Количество кадров на которые смотрим вперед от текущего
    # Проходим по всем детекторам полосы
    for i in range(0, len(detectors)):
        for j in range(0, frameCounter - 1):
            detector_activations_count = 0  # Счетчик активаций детектора
            # Смотрим на 1..frames_unite кадров вперед от текущего
            for k in range(1, frames_unite):
                # Если детектор активирован на текущем и 1..frames_unite кадре
                if (detectors[i].detections[j] == 1 and ((j + k) < (frameCounter - k - 1)) and (
                        detectors[i].detections[j + k] == 1)):
                    # Увеличиваем счетчик срабатываний детектора
                    detector_activations_count += 1
                # Вручную активируем детектор на detector_activations_count кадров от текущего
                for l in range(j, j + detector_activations_count):
                    detectors[i].detections[l] = 1

    #  Еще раз проходим по всем детекторам полосы
    for i in range(0, len(detectors)):
        # Смотрим значения активации детекторов для предыдущего и следующего кадров
        for j in range(0, frameCounter - 2):
            neighbours_sum = detectors[i].detections[j - 1] + detectors[i].detections[j + 1]
            # Если активации отсутствуют, то деактивируем детектор на текущем кадре
            if detectors[i].detections[j] == 1 and neighbours_sum == 0:
                detectors[i].detections[j] = 0


########################################################################################################################
# Дискретизация активаций детекторов
def detectors_discretization(detectors, frameCounter):
    # Значение разницы среднего цвета в % при котором будет проходить активация детектора
    activation_avg_colour_delta = 1.5
    # Задаем начальное значение активации всех детекторов на всех кадрах как 0
    for detector in detectors:
        detector.detections = [0] * frameCounter

    # Проходимся по всем кадрам детектора
    for i in range(0, len(detectors)):
        for j in range(0, frameCounter - 1):
            # Считаем текущую разницу среднего цвета между текущим и предыдущим кадрами
            current_avg_colour_delta = abs(
                (detectors[i].avgColour[j] - detectors[i].avgColour[j + 1]) / detectors[i].avgColour[j]) * 100
            # Если разница больше заданной, то активируем детектор
            if current_avg_colour_delta > activation_avg_colour_delta:
                detectors[i].detections[j + 1] = 1


########################################################################################################################
# Получение среднего значения цвета
def get_avg_colour_sum(gray, detectors):
    # для всех детекторов
    for detector in detectors:
        # Вырезаем область детектора
        detector_zone = gray[int((detector.detY - (size / 2))):int((detector.detY + (size / 2))),
                        int((detector.detX - (size / 2))):int((detector.detX + (size / 2)))]
        avg_color_per_row = np.average(detector_zone, axis=0)  # считаем средние цвета для пикселей по горизонтали
        avg_color = np.average(avg_color_per_row, axis=0)  # считаем средний цвет для средних цветов по горизонтали
        # Добавляем значение среднего цвета для детектора
        detector.add_avg_colour_sum(avg_color)
        #print(avg_color)  # Выводм значение среднего цвета для детектора в консоль

        # Отображаем окно для проверки детектора
        show = False
        if show:
            cv.namedWindow('detector', cv.WINDOW_NORMAL)
            cv.imshow('detector', detector_zone)


########################################################################################################################
# Отрисовка детектора
def draw_detector(x, y, lane_number):
    # Набор доступных цветов
    colours = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 127], [0, 127, 0], [127, 0, 0]]
    cv.rectangle(frame, (int(x - (size / 2)), int(y - (size / 2))),
                 (int(x + (size / 2)), int(y + (size / 2))), colours[lane_number], 2)


########################################################################################################################
# Установка детектора
def set_detector(event, x, y, flags, param, count_of_lanes=0):
    global mouseX, mouseY, drawing

    # Если нажата левая кнопка мыши
    if event == cv.EVENT_LBUTTONDOWN:
        # Проверяем не достигнуто ли максимальное количество детекторов на полосу
        if len(lanes) != 0 and len(lanes) * len(lanes[0]) == number_of_lanes * detectors_per_lane:
            print("Max number of detectors reached")
        else:
            # Получаем координаты клика
            mouseX, mouseY = x, y
            # Отрисовываем детектор по координатам
            draw_detector(mouseX, mouseY, len(lanes))
            # Обновляем окно с кадром
            cv.imshow('Frame', frame)
            # Выводим координаты цента детектора в консоль
            print(str(mouseX) + " " + str(mouseY))
            # Добавляем детектор в список детекторов полосы
            if len(detectors) < detectors_per_lane:
                detectors.append(Detector(mouseX, mouseY))
            # Добавляем полосу в список полос
            if len(detectors) == detectors_per_lane:
                lanes.append(detectors.copy())
                # Очищаем текущий список детекторов
                detectors.clear()


########################################################################################################################
# Открытие видео
cap = cv.VideoCapture('test.mp4')  # Видеофайл для обработки

# Проверка открытия файла
if not cap.isOpened():
    print("Error opening file")

########################################################################################################################
# Отображение и расстановка детекторов для первого кадра

set_detectors = False # Будет ли ручная расстановка детекторов

if set_detectors:
    # Создание окна для отображения первого кадра
    cv.namedWindow('Frame', cv.WINDOW_NORMAL)
    # Установка обработчика событий для кадра
    cv.setMouseCallback('Frame', set_detector)

    # Установка новых размеров кадра
    cv.resizeWindow('Frame', 1600, 800)

    # Считывание, отображение и расстановка детекторов для первого кадра
    ret, frame = cap.read()
    cv.imshow('Frame', frame)
    cv.waitKey(0)
else:
    df = pd.read_csv('Coordinates.csv')
    if df.empty:
        print('Coordinates.csv DataFrame is empty!')
    for i in range(len(df)):
        coordinates = df.values[i][0].split(';')

    coordinates_counter = 0
    for i in range(0, number_of_lanes*detectors_per_lane):
        if len(detectors) < detectors_per_lane:
            detectors.append(Detector(int(coordinates[coordinates_counter]), int(coordinates[coordinates_counter + 1])))
            coordinates_counter += 2
        if len(detectors) == detectors_per_lane:
            lanes.append(detectors.copy())
        # Очищаем текущий список детекторов
            detectors.clear()

########################################################################################################################

video_path = 'Videos/'
videos = os.listdir(video_path)

# Обработка видео
cv.namedWindow('Frame', cv.WINDOW_NORMAL)
cv.resizeWindow('Frame', 800, 600)

frame_counter = 0  # Счетчик кадров

# Обработка видео до последнего кадра
# Пока поступают кадры видео

for video in videos:
    cap = cv.VideoCapture(video)
    while cap.isOpened():
        # Считываем очередной кадр
        ret, frame = cap.read()
        # Проверяем открытие кадра
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Переводим кадр в градации серого
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Если считался очередной кадр
        if ret:
            # Выводим номер кадра в верхнем левом углу
            cv.putText(frame, "Frame " + str(frame_counter), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                       cv.LINE_AA)
            # Начинаем отрисовку детекторов
            colour_counter = 0  # счетчк задействованных цветов
            # Отрисовываем детекторы для каждой полосы
            for lane in lanes:
                for detector in lane:
                    draw_detector(detector.detX, detector.detY, colour_counter)
                colour_counter += 1
                # Считаем среднее значение цвета для каждого детектора
                get_avg_colour_sum(gray, lane)
            cv.imshow('Frame', frame)
            # if frameCounter % 10 == 0:
            # cv.imwrite("Frame_" + str(frameCounter) + ".png",frame)
            # Увеличиваем счетчик кадров
            frame_counter += 1
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    # Освобождаем видео и удаляем все окна
    cap.release()
    cv.destroyAllWindows()


########################################################################################################################
# Заполняем csv файл координатами детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber) + " X": detector.detX}
        data.update(new_dict)
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber) + " Y": detector.detY}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame([data])
df.to_csv(r'Coordinates.csv', sep=';', index=False)

########################################################################################################################
# Заполняем csv файл средними цветами с детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber): detector.avgColour}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame(data)
df.to_csv(r'AvgColours.csv', sep=';', index=False)

########################################################################################################################
# Дискретизируем значения детектров
for lane in lanes:
    detectors_discretization(lane, frame_counter)

########################################################################################################################
# Заполняем csv файл дискретными значениями с детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber): detector.detections}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame(data)
df.to_csv(r'RawDetections.csv', sep=';', index=False)

########################################################################################################################
# Фильтруем дискретные значения детектров
for lane in lanes:
    detectors_discretization_filter(lane, frame_counter)

########################################################################################################################
# Заполняем csv файл дискретными отфильтрованными значениями с детекторов
data = dict()
lane_counter = 1
for lane in lanes:
    detectorNumber = 1
    for detector in lane:
        new_dict = {"lane" + str(lane_counter) + " det" + str(detectorNumber): detector.detections}
        data.update(new_dict)
        detectorNumber += 1
    lane_counter += 1
df = pd.DataFrame(data)
df.to_csv(r'FilteredDetections.csv', sep=';', index=False)

########################################################################################################################

# Создаем массив значений по оси x в интервале 1..frame_counter
x = []
for i in range(1, frame_counter + 1):
    x.append(i)

########################################################################################################################
# Построение графиков средних цветов для детекторов
position = 1  # Позиция субграфика
# Создаем субграфики для каждого детектора
fig, axs = plt.subplots(number_of_lanes, detectors_per_lane)
i = 0
for lane in lanes:
    j = 0
    for detector in lane:
        axs[i, j].plot(x, detector.avgColour)
        axs[i, j].set_title("Lane №" + str(i) + "detector №" + str(j))
        j += 1
    i += 1
for ax in axs.flat:
    ax.set(xlabel="Frames", ylabel="Average colour")
# Отображаем графики средних цветов для детекторов
plt.show()

########################################################################################################################
# Построение графиков активаций для детекторов
fig, axs = plt.subplots(number_of_lanes, detectors_per_lane)
i = 0
for lane in lanes:
    j = 0
    for detector in lane:
        axs[i, j].plot(x, detector.detections)
        axs[i, j].set_title("Lane №" + str(i) + "detector №" + str(j))
        j += 1
    i += 1
for ax in axs.flat:
    ax.set(xlabel="Frames", ylabel="Detections")
plt.show()

########################################################################################################################
