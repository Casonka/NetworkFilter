# System includes
import os
# ------------------------- #
# TensorFlow includes
import tensorflow as tf
from tensorflow import keras
from keras import layers
# ------------------------- #
# Plot library
import matplotlib.pyplot as plt
import matplotlib
# ------------------------- #
# Data frames libraries
import numpy as np
import pandas as pd

# ------------------------- #
# Dataset Files
FILEPATH = "datasets/data_20.0.csv"
# FILEPATH = "datasets/data_30.0.csv"
# FILEPATH = "datasets/data_40.0.csv"
# FILEPATH = "datasets/data_50.0.csv"
# ------------------------- #
# Item group variable
ITEMS = 500
DEF_K = 0.6
MEDIAN_WINDOW_SIZE = 4
MOVING_AVERAGE_SIZE = 4


# ------------------------- #
# Filter defs
def moving_average_filter_calc(data, window):
    result = np.zeros((len(data), 1))
    for i in range(len(data) - 1):
        for j in range(window):
            result[i] = result[i] + data[(i*window) + j]
        result[i] = result[i] / window
    return result


def low_pass_filter_calc(data, k):
    result = np.zeros((len(data), 1))
    point = 0
    for i in data:
        if point == 0:
            result[point] = (1 - k) * data[point]
        else:
            result[point] = (1 - k) * data[point] + k * data[point-1]
        point = point + 1
    return result


def moving_average_demo(accel, time):
    accel_f1 = moving_average_filter_calc(accel['accelY'], 6)
    plt.subplot(3, 1, 1)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f1, color="orange", linewidth=2)
    plt.title("K = 0.3")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f2 = moving_average_filter_calc(accel['accelY'], 12)
    plt.subplot(3, 1, 2)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f2, color="orange", linewidth=2)
    plt.title("K = 0.6")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f3 = moving_average_filter_calc(accel['accelY'], 18)
    plt.subplot(3, 1, 3)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f3, color="orange", linewidth=2)
    plt.title("K = 0.8")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()
    plt.tight_layout(h_pad=0.01)
    plt.show()

def low_pass_filter_demo(accel, time):
    accel_f1 = low_pass_filter_calc(accel['accelY'], 0.3)
    plt.subplot(3, 1, 1)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f1, color="orange", linewidth=2)
    plt.title("K = 0.3")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f2 = low_pass_filter_calc(accel['accelY'], 0.6)
    plt.subplot(3, 1, 2)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f2, color="orange", linewidth=2)
    plt.title("K = 0.6")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f3 = low_pass_filter_calc(accel['accelY'], 0.8)
    plt.subplot(3, 1, 3)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f3, color="orange", linewidth=2)
    plt.title("K = 0.8")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()
    plt.tight_layout(h_pad=0.01)
    plt.show()


def median_filter_sort(data):
    for i in range(len(data) - 1):
        for j in range(len(data) - i - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data


def median_filter_calc(data, window):
    filtered = np.zeros((len(data)))
    for i in range(len(data)):
        if len(data) - i < window:
            tmp = data[i:(i+(len(data) - i))]
        else:
            tmp = data[i:(i + window)]
        tmp = tmp.sort_values(ascending=True)
        if len(tmp) < 2:
            filtered[i] = tmp
        else:
            filtered[i] = tmp.iloc[len(tmp) // 2]
    return filtered


def median_filter_demo(accel, time):
    accel_f1 = median_filter_calc(accel['accelX'], 7)
    plt.subplot(3, 1, 1)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f1, color="orange", linewidth=2)
    plt.title("Окно - 7 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f2 = median_filter_calc(accel['accelY'], 12)
    plt.subplot(3, 1, 2)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f2, color="orange", linewidth=2)
    plt.title("Окно 12 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f3 = median_filter_calc(accel['accelY'], 18)
    plt.subplot(3, 1, 3)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f3, color="orange", linewidth=2)
    plt.title("Окно 18 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()
    plt.tight_layout(h_pad=0.01)
    plt.show()


def median_filter_show(accel, gyro, time):
    plt.suptitle("Median Filter")

    accel_fx = median_filter_calc(accel['accelX'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 1)
    plt.plot(time, accel['accelX'], color="red", linewidth=3)
    plt.plot(time, accel_fx, color="orange", linewidth=2)
    plt.title("Accelerometer X")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fy = median_filter_calc(accel['accelY'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 3)
    plt.plot(time, accel['accelY'], color="green")
    plt.plot(time, accel_fy, color="orange")
    plt.title("Accelerometer Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fz = median_filter_calc(accel['accelZ'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 5)
    plt.plot(time, accel['accelZ'], color="purple")
    plt.plot(time, accel_fz, color="orange")
    plt.title("Accelerometer Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fx = median_filter_calc(gyro['gyroX'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 2)
    plt.plot(time, gyro['gyroX'], color="red")
    plt.plot(time, gyro_fx, color="orange")
    plt.title("Gyroscope X")
    plt.xlabel("Время, сек")
    plt.ylabel("Угловая скорость, ")
    plt.grid()

    gyro_fy = median_filter_calc(gyro['gyroY'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 4)
    plt.plot(time, gyro['gyroY'], color="green")
    plt.plot(time, gyro_fy, color="orange")
    plt.title("Gyroscope Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fz = median_filter_calc(gyro['gyroZ'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 6)
    plt.plot(time, gyro['gyroZ'], color="purple")
    plt.plot(time, gyro_fz, color="orange")
    plt.title("Gyroscope Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    plt.tight_layout(h_pad=0.1)
    plt.show()


def low_pass_filter_show(accel, gyro, time):
    plt.suptitle("Low Pass Filter")

    accel_fx = low_pass_filter_calc(accel['accelX'], DEF_K)
    plt.subplot(3, 2, 1)
    plt.plot(time, accel['accelX'], color="red", linewidth=3)
    plt.plot(time, accel_fx, color="orange", linewidth=2)
    plt.title("Accelerometer X")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fy = low_pass_filter_calc(accel['accelY'], DEF_K)
    plt.subplot(3, 2, 3)
    plt.plot(time, accel['accelY'], color="green")
    plt.plot(time, accel_fy, color="orange")
    plt.title("Accelerometer Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fz = low_pass_filter_calc(accel['accelZ'], DEF_K)
    plt.subplot(3, 2, 5)
    plt.plot(time, accel['accelZ'], color="purple")
    plt.plot(time, accel_fz, color="orange")
    plt.title("Accelerometer Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fx = low_pass_filter_calc(gyro['gyroX'], DEF_K)
    plt.subplot(3, 2, 2)
    plt.plot(time, gyro['gyroX'], color="red")
    plt.plot(time, gyro_fx, color="orange")
    plt.title("Gyroscope X")
    plt.xlabel("Время, сек")
    plt.ylabel("Угловая скорость, ")
    plt.grid()

    gyro_fy = low_pass_filter_calc(gyro['gyroY'], DEF_K)
    plt.subplot(3, 2, 4)
    plt.plot(time, gyro['gyroY'], color="green")
    plt.plot(time, gyro_fy, color="orange")
    plt.title("Gyroscope Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fz = low_pass_filter_calc(gyro['gyroZ'], DEF_K)
    plt.subplot(3, 2, 6)
    plt.plot(time, gyro['gyroZ'], color="purple")
    plt.plot(time, gyro_fz, color="orange")
    plt.title("Gyroscope Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    plt.tight_layout(h_pad=0.1)
    plt.show()


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    file = pd.read_csv(FILEPATH)
    timestamp = file['time'].head(ITEMS)
    accelerometer = file[['accelX', 'accelY', 'accelZ']].head(ITEMS)
    gyroscope = file[['gyroX', 'gyroY', 'gyroZ']].head(ITEMS)
    # low_pass_filter_show(accelerometer, gyroscope, timestamp)
    # low_pass_filter_demo(accelerometer, timestamp)
    # median_filter_show(accelerometer, gyroscope, timestamp)
    # median_filter_demo(accelerometer, timestamp)
    moving_average_demo(accelerometer, timestamp)