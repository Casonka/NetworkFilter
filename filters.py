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
FILEPATH = "datasets/data_12.0.csv"
# FILEPATH = "datasets/data_30.0.csv"
# FILEPATH = "datasets/data_40.0.csv"
# FILEPATH = "datasets/data_50.0.csv"
# ------------------------- #
# Item group variable
ITEMS = 10000
DEF_K = 0.49
MEDIAN_WINDOW_SIZE = 3
MOVING_AVERAGE_SIZE = 15


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


def moving_average_demo(datafile):
    accel_fX = moving_average_filter_calc(datafile['accelX'], MOVING_AVERAGE_SIZE)
    accel_fY = moving_average_filter_calc(datafile['accelY'], MOVING_AVERAGE_SIZE)
    gyro_fZ = moving_average_filter_calc(datafile['gyroZ'], MOVING_AVERAGE_SIZE)
    time = datafile.iloc[:, 0]

    real_coordX = datafile.iloc[:, 11]
    real_coordY = datafile.iloc[:, 12]
    coords = np.zeros([len(datafile), 4])
    velocityX, velocityY = 0, 0
    for i in range(len(coords)):
        real_angle = datafile.iloc[i, 7]

        prev_coordX = datafile.iloc[i, 11]
        prev_coordY = datafile.iloc[i, 12]
        dx, dy = navigation.calc_delta_from_accel_gyro(0.05, accel_fX[i], accel_fY[i], gyro_fZ[i], real_angle,
                                                       coords[i, 0], coords[i, 1], velocityX, velocityY)

        newX = prev_coordX + dx
        newY = prev_coordY + dy

        errorX = newX - datafile.iloc[i, 11]
        errorY = newY - datafile.iloc[i, 12]

        coords[i, 0] = newX
        coords[i, 1] = newY

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
        real_angle = datafile.iloc[i, 7]

        prev_coordX = datafile.iloc[i, 11]
        prev_coordY = datafile.iloc[i, 12]
        dx, dy = navigation.calc_delta_from_accel_gyro(0.05, accel_fX[i], accel_fY[i], gyro_fZ[i], real_angle,
                                                               coords[i, 0], coords[i, 1], velocityX, velocityY)

        newX = prev_coordX + dx
        newY = prev_coordY + dy

        errorX = newX - datafile.iloc[i, 11]
        errorY = newY - datafile.iloc[i, 12]

        coords[i, 0] = newX
        coords[i, 1] = newY

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
    velocityX, velocityY = 0, 0
    for i in range(len(coords)):
        real_angle = datafile.iloc[i, 7]

        prev_coordX = datafile.iloc[i, 11]
        prev_coordY = datafile.iloc[i, 12]
        dx, dy = navigation.calc_delta_from_accel_gyro(0.05, accel_fX[i], accel_fY[i], gyro_fZ[i], real_angle,
                                                       coords[i, 0], coords[i, 1], velocityX, velocityY)

        newX = prev_coordX + dx
        newY = prev_coordY + dy

        errorX = newX - datafile.iloc[i, 11]
        errorY = newY - datafile.iloc[i, 12]

        coords[i, 0] = newX
        coords[i, 1] = newY

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


if __name__ == '__main__':
    file = pd.read_csv(FILEPATH)
    # timestamp = file['time'].head(ITEMS)
    # accelerometer = file[['accelX', 'accelY', 'accelZ']].head(ITEMS)
    # gyroscope = file[['gyroX', 'gyroY', 'gyroZ']].head(ITEMS)
    # low_pass_filter_show(accelerometer, gyroscope, timestamp)
    # low_pass_filter_demo(file)
    # median_filter_show(accelerometer, gyroscope, timestamp)
    # median_filter_demo(file)
    moving_average_demo(file)
