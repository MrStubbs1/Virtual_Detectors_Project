from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from GUI_detectors import Ui_MainWindow  # импорт нашего сгенерированного файла
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtGui import QIcon, QPixmap
import cv2 as cv
import sys
import pandas as pd
import os
from Calculations import *
from FigureBuilder import *

########################################################################################################################
drawing = False
mouseX, mouseY = -1, -1  # Переменные для хранения координат курсора мыши
lanes = []  # Список полос
detectors = []  # Список детекторов


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
# Заполняем csv файл координатами детекторов
def fill_detector_coordinates():
    data = dict()
    lane_counter = 1
    for lane in lanes:
        detector_number = 1
        for detector in lane:
            new_dict = {"X " + str(lane_counter) + "." + str(detector_number): detector.detX}
            data.update(new_dict)
            new_dict = {"Y " + str(lane_counter) + "." + str(detector_number): detector.detY}
            data.update(new_dict)
            detector_number += 1
        lane_counter += 1
    df = pd.DataFrame([data])
    df.to_csv(r'Coordinates.csv', sep=';', index=False)


########################################################################################################################
# Заполняем csv файл координатами детекторов
def fill_detector_parameter(height, width, activation, filter):
    data = dict()
    lane_counter = 1
    new_dict = {"width": width, "height": height, "activation": activation, "filter": filter}
    data.update(new_dict)
    df = pd.DataFrame([data])
    df.to_csv(r'Detector_parameter.csv', sep=';', index=False)


########################################################################################################################
# Отображение и расстановка детекторов для первого кадра
def set_detectors_by_hand(frame):
    # Создание окна для отображения первого кадра
    cv.namedWindow('Frame', cv.WINDOW_NORMAL)
    # Установка обработчика событий для кадра
    cv.setMouseCallback('Frame', set_detector)

    # Установка новых размеров кадра
    cv.resizeWindow('Frame', 1600, 800)

    # Считывание, отображение и расстановка детекторов для первого кадра
    cv.imshow('Frame', frame)
    cv.waitKey(0)


########################################################################################################################
# Установка детектора
def set_detector(event, x, y, flags, param, count_of_lanes=0):
    global mouseX, mouseY, drawing

    # Если нажата левая кнопка мыши
    if event == cv.EVENT_LBUTTONDOWN:
        # Проверяем не достигнуто ли максимальное количество детекторов на полосу
        if len(lanes) != 0 and len(lanes) * len(lanes[0]) == number_of_lanes * detectors_per_lane:
            print("Max number of detectors reached")
            cv.putText(frame, "Max number of detectors", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 255, 255), 2,
                       cv.LINE_AA)
            cv.imshow('Frame', frame)
        else:
            # Получаем координаты клика
            mouseX, mouseY = x, y
            # Отрисовываем детектор по координатам
            draw_detector(frame, mouseX, mouseY, len(lanes), detector_height, detector_width)
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
# Отрисовка детектора
def draw_detector(frame, x, y, lane_number, width, height):
    # Набор доступных цветов
    colours = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 127], [0, 127, 0], [127, 0, 0]]
    cv.rectangle(frame, (int(x - (width / 2)), int(y - (width / 2))),
                 (int(x + (height / 2)), int(y + (height / 2))), colours[lane_number], 2)


########################################################################################################################
# Получение среднего значения цвета
def get_avg_colour_sum(detector, gray, lane, detector_width, detector_height):
    # для всех детекторов
    # Вырезаем область детектора
    detector_zone = gray[int((detector.detY - (detector_height / 2))):int((detector.detY + (detector_height / 2))),
                    int((detector.detX - (detector_width / 2))):int((detector.detX + (detector_width / 2)))]
    avg_color_per_row = np.average(detector_zone, axis=0)  # считаем средние цвета для пикселей по горизонтали
    avg_color = np.average(avg_color_per_row, axis=0)  # считаем средний цвет для средних цветов по горизонтали
    # Добавляем значение среднего цвета для детектора
    return  avg_color
    # print(avg_color)  # Выводм значение среднего цвета для детектора в консоль


########################################################################################################################
# Заполняем csv файл средними цветами с детекторов
def fill_avg_colours():
    data = dict()
    lane_counter = 1
    for lane in lanes:
        detector_number = 1
        for detector in lane:
            new_dict = {"lane" + str(lane_counter) + " det" + str(detector_number): detector.avgColour}
            data.update(new_dict)
            detector_number += 1
        lane_counter += 1
    df = pd.DataFrame(data)
    df.to_csv(r'AvgColours.csv', sep=';', index=False)


########################################################################################################################
# Дискретизируем значения детекторов
def start_discretisation():
    for lane in lanes:
        lane = detectors_discretization(lane, detector_activation)
    fill_activations()


########################################################################################################################
# Заполняем csv файл дискретными значениями с детекторов
def fill_activations():
    data = dict()
    lane_counter = 1
    for lane in lanes:
        detector_number = 1
        for detector in lane:
            new_dict = {"lane" + str(lane_counter) + " det" + str(detector_number): detector.detections}
            data.update(new_dict)
            detector_number += 1
        lane_counter += 1
    df = pd.DataFrame(data)
    df.to_csv(r'RawDetections.csv', sep=';', index=False)


########################################################################################################################
# Фильтруем дискретные значения детекторов
def start_filtering_activations():
    for lane in lanes:
        detectors_discretization_filter(lane, detector_filter)
    fill_filtered_activations()


########################################################################################################################
# Заполняем csv файл дискретными отфильтрованными значениями с детекторов
def fill_filtered_activations():
    data = dict()
    lane_counter = 1
    for lane in lanes:
        detector_number = 1
        for detector in lane:
            new_dict = {"lane" + str(lane_counter) + " det" + str(detector_number): detector.detections}
            data.update(new_dict)
            detector_number += 1
        lane_counter += 1
    df = pd.DataFrame(data)
    df.to_csv(r'FilteredDetections.csv', sep=';', index=False)


########################################################################################################################
class DetectorsGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(DetectorsGUI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Выбор типа расстановки детекторов
        if self.ui.rbn_set_detectors_from_file.isChecked():
            self.ui.number_of_lanes.setReadOnly(True)
            self.ui.detectors_per_lane.setReadOnly(True)

        ################################################################################################################
        # Привязка методов к кнопкам
        self.ui.rbn_set_detectors_by_hand.toggled.connect(self.set_detectors_label)
        self.ui.rbn_set_detectors_from_file.toggled.connect(self.set_detectors_label)
        self.ui.btn_set_detectors.clicked.connect(self.set_detectors)
        self.ui.btn_start_detection.clicked.connect(self.start_detection)
        self.ui.btn_build_graph.clicked.connect(self.build_graph)

    ####################################################################################################################
    # Отключение полей ввода при выборе ввода данных из файла
    def set_detectors_label(self):
        if self.ui.rbn_set_detectors_by_hand.isChecked():
            self.ui.number_of_lanes.setReadOnly(False)
            self.ui.detectors_per_lane.setReadOnly(False)
            self.ui.detector_height_edit.setReadOnly(False)
            self.ui.detector_width_edit.setReadOnly(False)
            self.ui.detector_activation_edit.setReadOnly(False)
            self.ui.detector_filter_edit.setReadOnly(False)
        if self.ui.rbn_set_detectors_from_file.isChecked():
            self.ui.number_of_lanes.setReadOnly(True)
            self.ui.detectors_per_lane.setReadOnly(True)
            self.ui.detector_height_edit.setReadOnly(True)
            self.ui.detector_width_edit.setReadOnly(True)
            self.ui.detector_activation_edit.setReadOnly(True)
            self.ui.detector_filter_edit.setReadOnly(True)

    ####################################################################################################################
    # Расстановка детекторов
    def set_detectors(self):

        global number_of_lanes  # количество полос
        global detectors_per_lane  # количество детекторов на полосу
        global detector_height  # высота детектора
        global detector_width  # ширина детектора
        global detector_activation  # порог активации детектора
        global detector_filter  # размер шага фильтра дискретизации
        # Если выбрана расстановка вручную
        if self.ui.rbn_set_detectors_by_hand.isChecked():
            # Проверки на заполнение данных
            if self.ui.number_of_lanes.text() == "":
                QMessageBox.critical(self, "Ошибка ", "Задайте данные по количеству полос", QMessageBox.Ok)
            elif self.ui.detectors_per_lane.text() == "":
                QMessageBox.critical(self, "Ошибка ", "Задайте данные по количеству детекторов", QMessageBox.Ok)
            elif self.ui.detector_height_edit.text() == "":
                QMessageBox.critical(self, "Ошибка ", "Задайте данные по высоте детектора", QMessageBox.Ok)
            elif self.ui.detector_width_edit.text() == "":
                QMessageBox.critical(self, "Ошибка ", "Задайте данные по ширине детектора", QMessageBox.Ok)
            elif self.ui.detector_activation_edit.text() == "":
                QMessageBox.critical(self, "Ошибка ", "Задайте данные по порогу активации детектора", QMessageBox.Ok)
            elif self.ui.detector_filter_edit.text() == "":
                QMessageBox.critical(self, "Ошибка ", "Задайте данные по размеру шага фильтра дискретизации",
                                     QMessageBox.Ok)
            else:
                number_of_lanes = int(self.ui.number_of_lanes.text())  # количество полос
                detectors_per_lane = int(self.ui.detectors_per_lane.text())  # количество детекторов на полосу
                detector_height = int(self.ui.detector_height_edit.text())  # высота детектора
                detector_width = int(self.ui.detector_width_edit.text())  # ширина детектора
                detector_activation = float(self.ui.detector_activation_edit.text())  # порог активации детектора
                detector_filter = int(self.ui.detector_filter_edit.text())  # размер шага фильтра дискретизации

                # Получаем список всех файлов в папке
                video_path = 'Videos/'
                files = os.listdir(video_path)
                videos = []

                # Добавляем в список только файлы с нужным расширением
                for file in files:
                    if file.endswith(".mp4"):
                        videos.append(file)

                # Открытие видео
                cap = cv.VideoCapture("Videos/" + videos.pop())  # Видеофайл для обработки
                # Проверка открытия файла
                if not cap.isOpened():
                    print("Error opening file")
                global frame
                ret, frame = cap.read()
                set_detectors_by_hand(frame)  # Расставляем детекторы вручную
                fill_detector_coordinates()  # Сохраняем координаты детекторов
                # Сохраняем параметры детекторов
                fill_detector_parameter(detector_height, detector_width, detector_activation, detector_filter)

        # Если выбрана расстановка из файла
        else:
            if os.path.exists('Coordinates.csv'):
                df = pd.read_csv('Coordinates.csv')
                # Проверяем не пустой ли файл
                if df.empty:
                    QMessageBox.critical(self, "Ошибка ", "Файл Coordinates.csv с координатами детекторов пустой",
                                         QMessageBox.Ok)
                else:
                    # Получаем координаты детекторов из файла
                    columns_list = df.columns.values.tolist()
                    buf = columns_list[0].split(';')
                    buf = buf[len(buf) - 1].split(' ')
                    buf = buf[1].split('.')
                    number_of_lanes = int(buf[0])
                    detectors_per_lane = int(buf[1])

                    for i in range(len(df)):
                        coordinates = df.values[i][0].split(';')

                    coordinates_counter = 0
                    for i in range(0, number_of_lanes * detectors_per_lane):
                        if len(detectors) < detectors_per_lane:
                            detectors.append(
                                Detector(int(coordinates[coordinates_counter]),
                                         int(coordinates[coordinates_counter + 1])))
                            coordinates_counter += 2
                        if len(detectors) == detectors_per_lane:
                            lanes.append(detectors.copy())
                            # Очищаем текущий список детекторов
                            detectors.clear()
            else:
                QMessageBox.critical(self, "Ошибка ", "Отсутствует файл Coordinates.csv с координатами детекторов",
                                     QMessageBox.Ok)
        if os.path.exists('Detector_parameter.csv'):
            df = pd.read_csv('Detector_parameter.csv')
            if df.empty:
                # Проверяем не пустой ли файл
                QMessageBox.critical(self, "Ошибка ", "Файл Detector_parameter.csv с параметрами детекторов пустой",
                                     QMessageBox.Ok)
            else:
                # Получаем параметры детекторов из файла
                values = df.values[0][0].split(';')
                detector_height = int(values[0])
                detector_width = int(values[1])
                detector_activation = float(values[2])
                detector_filter = int(values[3])

                self.ui.number_of_lanes.setText(str(number_of_lanes))
                self.ui.detectors_per_lane.setText(str(detectors_per_lane))
                self.ui.detector_height_edit.setText(str(detector_height))
                self.ui.detector_width_edit.setText(str(detector_width))
                self.ui.detector_activation_edit.setText(str(detector_activation))
                self.ui.detector_filter_edit.setText(str(detector_filter))

                dlg = QMessageBox(self)
                dlg.setWindowTitle("Успех!")
                dlg.setText("Детекторы успешно добавлены")
                button = dlg.exec()

                if button == QMessageBox.Ok:
                    print("OK!")
        else:
            QMessageBox.critical(self, "Ошибка ", "Отсутствует файл Detector_parameter.csv с параметрами детекторов",
                                 QMessageBox.Ok)

    ####################################################################################################################
    # Старт обработки видео
    def start_detection(self):

        # Получаем список всех файлов в папке
        video_path = 'Videos/'
        files = os.listdir(video_path)
        videos = []

        # Добавляем в список только файлы с нужным расширением
        for file in files:
            if file.endswith(".mp4"):
                videos.append("Videos/" + file)

        frame_counter = 0  # Счетчик кадров
        frame_step = 1  # Шаг кадров
        video_counter = 0  # Счетчик видеофайлов
        self.ui.total_files_edit.setText(str(len(videos)))  # Отображаем общее количество видеофайлов

        '''
        total_video_length = 0
        for video in videos:
            cap = cv.VideoCapture(video)
            property_id = int(cv.CAP_PROP_FRAME_COUNT)
            video_length = int(cv.VideoCapture.get(cap, property_id))
            total_video_length += video_length
        self.ui.total_frames_edit.setText(str(total_video_length - 1))
        '''

        # Обработка видео до последнего кадра
        # Пока поступают кадры видео
        for video in videos:
            frames_in_video_counter = 0
            cap = cv.VideoCapture(video)
            # Считаем количество кадров в видеофайле
            property_id = int(cv.CAP_PROP_FRAME_COUNT)
            video_length = int(cv.VideoCapture.get(cap, property_id))
            self.ui.total_frames_edit.setText(str(video_length - 1))  # Отображаем общее кадров в видеофайле
            self.ui.filename_edit.setText(video)  # Отображаем общее имя видеофайла
            self.ui.processed_files_edit.setText(str(video_counter))  # Отображаем количество обработанных видеофайлов
            if self.ui.checkBox_visualization.isChecked():
                cv.namedWindow(video, cv.WINDOW_NORMAL)
                cv.resizeWindow(video, 800, 600)
            while cap.isOpened():
                # Считываем очередной кадр
                ret, frame = cap.read()
                # Проверяем открытие кадра
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                if ret:
                    if (frame_counter % frame_step == 0):
                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Переводим кадр в градации серого
                        if self.ui.checkBox_visualization.isChecked():  # Если включена визуализация
                            # Выводим обработанное/общее количество кадров
                            cv.putText(frame, "Frame " + str(frames_in_video_counter) + " / " + str(video_length - 1),
                                       (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
                        colour_counter = 0
                        for lane in lanes:
                            for detector in lane:
                                if self.ui.checkBox_visualization.isChecked():
                                    # Отрисовываем детектор
                                    draw_detector(frame, detector.detX, detector.detY, colour_counter, detector_height,
                                                  detector_width)
                                # Считаем средний цвет
                                detector.add_avg_colour_sum(
                                    get_avg_colour_sum(detector, gray, lane, detector_width, detector_height))
                            colour_counter += 1
                        if self.ui.checkBox_visualization.isChecked():
                            cv.imshow(video, frame)
                        # Отображаем число обработанных кадров
                        self.ui.processed_frames_edit.setText(str(frames_in_video_counter))
                        frames_in_video_counter += frame_step
                    frame_counter += 1
                    if cv.waitKey(25) & 0xFF == ord('q'):
                        break
            cap.release()
            video_counter += 1
            if self.ui.checkBox_visualization.isChecked():
                cv.destroyWindow(video)
        self.ui.processed_files_edit.setText(str(video_counter))
        fill_avg_colours()
        start_discretisation()
        start_filtering_activations()
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Успех!")
        dlg.setText("Обработка завершена")
        button = dlg.exec()

        if button == QMessageBox.Ok:
            print("OK!")

    ####################################################################################################################
    def build_graph(self):
        # Получаем количество полос и детекторов из файла с координатами
        if os.path.exists('Coordinates.csv'):
            df = pd.read_csv('Coordinates.csv')
            # Проверяем не пустой ли файл
            if df.empty:
                QMessageBox.critical(self, "Ошибка ", "Файл Coordinates.csv с координатами детекторов пустой",
                                     QMessageBox.Ok)
            else:
                # Получаем координаты детекторов из файла
                columns_list = df.columns.values.tolist()
                buf = columns_list[0].split(';')
                buf = buf[len(buf) - 1].split(' ')
                buf = buf[1].split('.')
                number_of_lanes = int(buf[0])
                detectors_per_lane = int(buf[1])

        if self.ui.time_slice_edit.text() == "":
            QMessageBox.critical(self, "Ошибка ", "Задайте временной интервал", QMessageBox.Ok)
        else:
            # Строим график плотности
            if self.ui.density_checkBox.isChecked():
                time_slice = int(self.ui.time_slice_edit.text())  # количество полос
                buf_lanes = add_activations_from_file(number_of_lanes, detectors_per_lane, 'FilteredDetections.csv')
                buf_lanes = cut_data(buf_lanes, time_slice)
                time_slice = cut_time_slice(buf_lanes, time_slice)
                density_lanes = get_density(buf_lanes)
                density_lanes = density_to_sec(density_lanes)
                fill_density(density_lanes)
                draw_density(density_lanes, number_of_lanes)

            if self.ui.intensity_checkBox.isChecked():
                time_slice = int(self.ui.time_slice_edit.text())  # количество полос
                buf_lanes = add_activations_from_file(number_of_lanes, detectors_per_lane, 'FilteredDetections.csv')
                buf_lanes = cut_data(buf_lanes, time_slice)
                time_slice = cut_time_slice(buf_lanes, time_slice)
                detections_per_lanes = get_activations_per_time_slice(buf_lanes, time_slice)
                fill_activations_per_time_slice(detections_per_lanes)
                intensity_per_lanes = get_intensity(detections_per_lanes)
                draw_intensity(intensity_per_lanes, number_of_lanes)

            if not self.ui.density_checkBox.isChecked() and not self.ui.intensity_checkBox.isChecked() \
                    and not self.ui.velocity_checkBox.isChecked():
                QMessageBox.critical(self, "Ошибка ", "Не выбран ни один из видов графиков", QMessageBox.Ok)


app = QtWidgets.QApplication([])
application = DetectorsGUI()
application.show()

sys.exit(app.exec())
