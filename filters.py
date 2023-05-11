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
MEDIAN_WINDOW_SIZE = 9
AVERAGE_WINDOW_SIZE = 10


# ------------------------- #
# Filter defs
def low_pass_filter_calc(data, K):
    result = np.zeros((len(data), 1))
    point = 0
    for i in data:
        if point == 0:
            result[point] = (1 - K) * data[point]
        else:
            result[point] = (1 - K) * data[point] + K * data[point - 1]
        point = point + 1
    return result


def low_pass_filter_demo(accel, time):
    accel_f1 = low_pass_filter_calc(accel['gyroY'], 0.3)
    plt.subplot(3, 1, 1)
    plt.plot(time, accel['gyroY'], color="green", linewidth=3)
    plt.plot(time, accel_f1, color="orange", linewidth=2)
    plt.title("K = 0.3")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f2 = low_pass_filter_calc(accel['gyroY'], 0.6)
    plt.subplot(3, 1, 2)
    plt.plot(time, accel['gyroY'], color="green", linewidth=3)
    plt.plot(time, accel_f2, color="orange", linewidth=2)
    plt.title("K = 0.6")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f3 = low_pass_filter_calc(accel['gyroY'], 0.8)
    plt.subplot(3, 1, 3)
    plt.plot(time, accel['gyroY'], color="green", linewidth=3)
    plt.plot(time, accel_f3, color="orange", linewidth=2)
    plt.title("K = 0.8")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()
    plt.tight_layout(h_pad=0.01)
    plt.show()


def moving_average_filter_calc(data, window):
    filtered = np.zeros(len(data))
    for i in range(len(data)):
        tmp = data[i:(i + window)]
        result = 0
        for j in tmp:
            result = result + j
        result = result / len(tmp)
        filtered[i] = result
    return filtered


def moving_average_filter_demo(accel, time):
    accel_f1 = moving_average_filter_calc(accel['accelY'], 6)
    plt.subplot(3, 1, 1)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f1, color="orange", linewidth=2)
    plt.title("Окно 6 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f2 = moving_average_filter_calc(accel['accelY'], 12)
    plt.subplot(3, 1, 2)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f2, color="orange", linewidth=2)
    plt.title("Окно 12 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_f3 = moving_average_filter_calc(accel['accelY'], 18)
    plt.subplot(3, 1, 3)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_f3, color="orange", linewidth=2)
    plt.title("Окно 18 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()
    plt.tight_layout(h_pad=0.01)
    plt.show()


def moving_average_filter_show(accel, gyro, time):
    plt.suptitle("Moving average filter")

    accel_fx = moving_average_filter_calc(accel['accelX'], AVERAGE_WINDOW_SIZE)
    plt.subplot(3, 2, 1)
    plt.plot(time, accel['accelX'], color="red", linewidth=3)
    plt.plot(time, accel_fx, color="orange", linewidth=2)
    plt.title("Accelerometer X")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fy = moving_average_filter_calc(accel['accelY'], AVERAGE_WINDOW_SIZE)
    plt.subplot(3, 2, 3)
    plt.plot(time, accel['accelY'], color="green")
    plt.plot(time, accel_fy, color="orange")
    plt.title("Accelerometer Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fz = moving_average_filter_calc(accel['accelZ'], AVERAGE_WINDOW_SIZE)
    plt.subplot(3, 2, 5)
    plt.plot(time, accel['accelZ'], color="purple")
    plt.plot(time, accel_fz, color="orange")
    plt.title("Accelerometer Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fx = moving_average_filter_calc(gyro['gyroX'], AVERAGE_WINDOW_SIZE)
    plt.subplot(3, 2, 2)
    plt.plot(time, gyro['gyroX'], color="red")
    plt.plot(time, gyro_fx, color="orange")
    plt.title("Gyroscope X")
    plt.xlabel("Время, сек")
    plt.ylabel("Угловая скорость, ")
    plt.grid()

    gyro_fy = moving_average_filter_calc(gyro['gyroY'], AVERAGE_WINDOW_SIZE)
    plt.subplot(3, 2, 4)
    plt.plot(time, gyro['gyroY'], color="green")
    plt.plot(time, gyro_fy, color="orange")
    plt.title("Gyroscope Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fz = moving_average_filter_calc(gyro['gyroZ'], AVERAGE_WINDOW_SIZE)
    plt.subplot(3, 2, 6)
    plt.plot(time, gyro['gyroZ'], color="purple")
    plt.plot(time, gyro_fz, color="orange")
    plt.title("Gyroscope Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    plt.tight_layout(h_pad=0.1)
    plt.show()


def median_filter_calc(data, window):
    filtered = np.zeros((len(data) // MEDIAN_WINDOW_SIZE))
    for i in range(len(data) // MEDIAN_WINDOW_SIZE):
        tmp = data[(i * MEDIAN_WINDOW_SIZE):((i * MEDIAN_WINDOW_SIZE) + MEDIAN_WINDOW_SIZE)]
        tmp = tmp.sort_values(ascending=True)
        filtered[i] = tmp.iloc[(MEDIAN_WINDOW_SIZE - 1) - 4]
    return filtered


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

    accel_fy = low_pass_filter_calc(accel['accelY'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 3)
    plt.plot(time, accel['accelY'], color="green")
    plt.plot(time, accel_fy, color="orange")
    plt.title("Accelerometer Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fz = low_pass_filter_calc(accel['accelZ'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 5)
    plt.plot(time, accel['accelZ'], color="purple")
    plt.plot(time, accel_fz, color="orange")
    plt.title("Accelerometer Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fx = low_pass_filter_calc(gyro['gyroX'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 2)
    plt.plot(time, gyro['gyroX'], color="red")
    plt.plot(time, gyro_fx, color="orange")
    plt.title("Gyroscope X")
    plt.xlabel("Время, сек")
    plt.ylabel("Угловая скорость, ")
    plt.grid()

    gyro_fy = low_pass_filter_calc(gyro['gyroY'], MEDIAN_WINDOW_SIZE)
    plt.subplot(3, 2, 4)
    plt.plot(time, gyro['gyroY'], color="green")
    plt.plot(time, gyro_fy, color="orange")
    plt.title("Gyroscope Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fz = low_pass_filter_calc(gyro['gyroZ'], MEDIAN_WINDOW_SIZE)
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
    # low_pass_filter_demo(gyroscope, timestamp)
    # median_filter_show(accelerometer, gyroscope, timestamp)
    # moving_average_filter_show(accelerometer, gyroscope, timestamp)
    # moving_average_filter_demo(accelerometer, timestamp)
