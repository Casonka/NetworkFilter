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

import navigation

# ------------------------- #
# Dataset Files
FILEPATH = "datasets/data_20.0.csv"
# FILEPATH = "datasets/data_30.0.csv"
# FILEPATH = "datasets/data_40.0.csv"
# FILEPATH = "datasets/data_50.0.csv"
# ------------------------- #
# Item group variable
ITEMS = 10000
DEF_K = 0.9
MEDIAN_WINDOW_SIZE = 12
MOVING_AVERAGE_SIZE = 10


# ------------------------- #
# Filter defs

def moving_average_filter_calc(data, window):
    result = np.zeros((len(data), 1))
    for i in range(len(data) - 1):
        for j in range(window):
            if (i * window) + j > (len(data) - 1):
                result[i] = result[i] + data[len(data) - 1]
            else:
                result[i] = result[i] + data[(i * window) + j]
        result[i] = result[i] / window
    return result


def low_pass_filter_calc(data, k):
    result = np.zeros((len(data), 1))
    point = 0
    for i in data:
        if point == 0:
            result[point] = (1 - k) * data[point]
        else:
            result[point] = (1 - k) * data[point] + k * data[point - 1]
        point = point + 1
    return result


def median_filter_calc(data, window):
    filtered = np.zeros((len(data)))
    for i in range(len(data)):
        if len(data) - i < window:
            tmp = data[i:(i + (len(data) - i))]
        else:
            tmp = data[i:(i + window)]
        tmp = tmp.sort_values(ascending=True)
        if len(tmp) < 2:
            filtered[i] = tmp
        else:
            filtered[i] = tmp.iloc[len(tmp) // 2]
    return filtered


def moving_average_demo(accel, time):
    accel_af1 = moving_average_filter_calc(accel['accelY'], 6)
    plt.subplot(3, 1, 1)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_af1, color="orange", linewidth=2)
    plt.title("Окно 6 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_af2 = moving_average_filter_calc(accel['accelY'], 12)
    plt.subplot(3, 1, 2)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_af2, color="orange", linewidth=2)
    plt.title("Окно 12 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_af3 = moving_average_filter_calc(accel['accelY'], 18)
    plt.subplot(3, 1, 3)
    plt.plot(time, accel['accelY'], color="green", linewidth=3)
    plt.plot(time, accel_af3, color="orange", linewidth=2)
    plt.title("Окно 18 элементов")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()
    plt.tight_layout(h_pad=0.01)
    plt.show()


def low_pass_filter_demo(datafile):
    accel_fX = low_pass_filter_calc(datafile['accelX'], DEF_K)
    accel_fY = low_pass_filter_calc(datafile['accelY'], DEF_K)
    gyro_fZ = low_pass_filter_calc(datafile['gyroZ'], DEF_K)
    time = datafile.iloc[:, 0]

    real_coordX = datafile.iloc[:, 11]
    real_coordY = datafile.iloc[:, 12]
    coords = np.zeros([len(datafile), 4])
    velocityX, velocityY = 0, 0
    for i in range(len(coords)):
        if i == 0:
            coords[i, 0] = datafile.iloc[i, 11]
            coords[i, 1] = datafile.iloc[i, 12]
            continue
        real_angle = datafile.iloc[i - 1, 7]

        # prev_coordX = datafile.iloc[i - 1, 11]
        # prev_coordY = datafile.iloc[i - 1, 12]
        fut_coordX = datafile.iloc[i, 11]
        fut_coordY = datafile.iloc[i, 12]
        coordX, coordY = navigation.calc_delta_from_accel_gyro(0.05, accel_fX[i], accel_fY[i], gyro_fZ[i], real_angle,
                                                               coords[i-1, 0], coords[i-1, 1], velocityX, velocityY)
        errorX = fut_coordX - coordX
        errorY = fut_coordY - coordY

        coords[i, 0] = coordX
        coords[i, 1] = coordY

        coords[i, 2] = errorX
        coords[i, 3] = errorY

    print(np.sqrt(np.square(np.subtract(real_coordX, coords[:, 0])).mean()))
    print(np.sqrt(np.square(np.subtract(real_coordY, coords[:, 1])).mean()))

    plt.subplot(2, 1, 1)
    plt.plot(time, real_coordX, color="green", linewidth=3)
    plt.plot(time, coords[:, 0], color="orange", linewidth=2)
    plt.title("K - " + str(DEF_K))
    plt.xlabel("Время, сек")
    plt.ylabel("Положение по Х")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, real_coordY, color="green", linewidth=3)
    plt.plot(time, coords[:, 1], color="orange", linewidth=2)
    plt.title("K - " + str(DEF_K))
    plt.xlabel("Время, сек")
    plt.ylabel("Положение по Y")
    plt.grid()
    plt.show()


def median_filter_demo(datafile):
    accel_fX = median_filter_calc(datafile['accelX'], MEDIAN_WINDOW_SIZE)
    accel_fY = median_filter_calc(datafile['accelY'], MEDIAN_WINDOW_SIZE)
    gyro_fZ = median_filter_calc(datafile['gyroZ'], MEDIAN_WINDOW_SIZE)
    time = datafile.iloc[:, 0]

    real_coordX = datafile.iloc[:, 11]
    real_coordY = datafile.iloc[:, 12]
    coords = np.zeros([len(datafile), 4])
    for i in range(len(coords)):
        if i == 0:
            coords[i, 0] = datafile.iloc[i, 11]
            coords[i, 1] = datafile.iloc[i, 12]
            continue
        real_angle = datafile.iloc[i - 1, 7]
        prev_coordX = datafile.iloc[i - 1, 11]
        prev_coordY = datafile.iloc[i - 1, 12]
        fut_coordX = datafile.iloc[i, 11]
        fut_coordY = datafile.iloc[i, 12]
        coordX, coordY = navigation.calc_delta_from_accel_gyro(0.05, accel_fX[i], accel_fY[i], gyro_fZ[i], real_angle,
                                                               coords[i-1, 0], coords[i-1, 1])
        errorX = fut_coordX - coordX
        errorY = fut_coordY - coordY

        coords[i, 0] = coordX
        coords[i, 1] = coordY

        coords[i, 2] = errorX
        coords[i, 3] = errorY

    print(np.sqrt(np.square(np.subtract(real_coordX, coords[:, 0])).mean()))
    print(np.sqrt(np.square(np.subtract(real_coordY, coords[:, 1])).mean()))

    plt.subplot(2, 1, 1)
    plt.plot(time, real_coordX, color="green", linewidth=3)
    plt.plot(time, coords[:, 0], color="orange", linewidth=2)
    plt.title("K - " + str(DEF_K))
    plt.xlabel("Время, сек")
    plt.ylabel("Положение по Х")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, real_coordY, color="green", linewidth=3)
    plt.plot(time, coords[:, 1], color="orange", linewidth=2)
    plt.title("K - " + str(DEF_K))
    plt.xlabel("Время, сек")
    plt.ylabel("Положение по Y")
    plt.grid()
    plt.show()


def moving_average_filter_show(accel, gyro, time):
    plt.suptitle("Moving average filter")

    accel_fx = moving_average_filter_calc(accel['accelX'], MOVING_AVERAGE_SIZE)
    plt.subplot(3, 2, 1)
    plt.plot(time, accel['accelX'], color="red", linewidth=3)
    plt.plot(time, accel_fx, color="orange", linewidth=2)
    plt.title("Accelerometer X")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fy = moving_average_filter_calc(accel['accelY'], MOVING_AVERAGE_SIZE)
    plt.subplot(3, 2, 3)
    plt.plot(time, accel['accelY'], color="green")
    plt.plot(time, accel_fy, color="orange")
    plt.title("Accelerometer Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_fz = moving_average_filter_calc(accel['accelZ'], MOVING_AVERAGE_SIZE)
    plt.subplot(3, 2, 5)
    plt.plot(time, accel['accelZ'], color="purple")
    plt.plot(time, accel_fz, color="orange")
    plt.title("Accelerometer Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fx = moving_average_filter_calc(gyro['gyroX'], MOVING_AVERAGE_SIZE)
    plt.subplot(3, 2, 2)
    plt.plot(time, gyro['gyroX'], color="red")
    plt.plot(time, gyro_fx, color="orange")
    plt.title("Gyroscope X")
    plt.xlabel("Время, сек")
    plt.ylabel("Угловая скорость, ")
    plt.grid()

    gyro_fy = moving_average_filter_calc(gyro['gyroY'], MOVING_AVERAGE_SIZE)
    plt.subplot(3, 2, 4)
    plt.plot(time, gyro['gyroY'], color="green")
    plt.plot(time, gyro_fy, color="orange")
    plt.title("Gyroscope Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_fz = moving_average_filter_calc(gyro['gyroZ'], MOVING_AVERAGE_SIZE)
    plt.subplot(3, 2, 6)
    plt.plot(time, gyro['gyroZ'], color="purple")
    plt.plot(time, gyro_fz, color="orange")
    plt.title("Gyroscope Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    plt.tight_layout(h_pad=0.1)
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

    accel_lx = low_pass_filter_calc(accel['accelX'], DEF_K)
    plt.subplot(3, 2, 1)
    plt.plot(time, accel['accelX'], color="red", linewidth=3)
    plt.plot(time, accel_lx, color="orange", linewidth=2)
    plt.title("Accelerometer X")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_ly = low_pass_filter_calc(accel['accelY'], DEF_K)
    plt.subplot(3, 2, 3)
    plt.plot(time, accel['accelY'], color="green")
    plt.plot(time, accel_ly, color="orange")
    plt.title("Accelerometer Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    accel_lz = low_pass_filter_calc(accel['accelZ'], DEF_K)
    plt.subplot(3, 2, 5)
    plt.plot(time, accel['accelZ'], color="purple")
    plt.plot(time, accel_lz, color="orange")
    plt.title("Accelerometer Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_lx = low_pass_filter_calc(gyro['gyroX'], DEF_K)
    plt.subplot(3, 2, 2)
    plt.plot(time, gyro['gyroX'], color="red")
    plt.plot(time, gyro_lx, color="orange")
    plt.title("Gyroscope X")
    plt.xlabel("Время, сек")
    plt.ylabel("Угловая скорость, ")
    plt.grid()

    gyro_ly = low_pass_filter_calc(gyro['gyroY'], DEF_K)
    plt.subplot(3, 2, 4)
    plt.plot(time, gyro['gyroY'], color="green")
    plt.plot(time, gyro_ly, color="orange")
    plt.title("Gyroscope Y")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    gyro_lz = low_pass_filter_calc(gyro['gyroZ'], DEF_K)
    plt.subplot(3, 2, 6)
    plt.plot(time, gyro['gyroZ'], color="purple")
    plt.plot(time, gyro_lz, color="orange")
    plt.title("Gyroscope Z")
    plt.xlabel("Время, сек")
    plt.ylabel("Ускорение, м/c^2")
    plt.grid()

    plt.tight_layout(h_pad=0.1)
    plt.show()


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    file = pd.read_csv(FILEPATH)
    # timestamp = file['time'].head(ITEMS)
    # accelerometer = file[['accelX', 'accelY', 'accelZ']].head(ITEMS)
    # gyroscope = file[['gyroX', 'gyroY', 'gyroZ']].head(ITEMS)
    # low_pass_filter_show(accelerometer, gyroscope, timestamp)
    low_pass_filter_demo(file)
    # median_filter_show(accelerometer, gyroscope, timestamp)
    # median_filter_demo(file)
    # moving_average_filter_calc(accelerometer, timestamp)
