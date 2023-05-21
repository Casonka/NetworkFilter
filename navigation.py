import math
import numpy as np
import pandas as pd

import filters
import matplotlib.pyplot as plt
import matplotlib


def get_start_pos(data):
    start_posX = data.iloc[0, 11]
    start_posY = data.iloc[0, 12]
    start_angle = data.iloc[0, 7] * 57.2958
    dt = data.iloc[1, 0] - data.iloc[0, 0]
    return start_posX, start_posY, start_angle, dt


def calc_position(data):
    accel = data.iloc[:, 1:3].values
    gyro = data.iloc[:, 6].values
    posX, posY, angle, dt = get_start_pos(data)
    result = np.zeros([len(data), 2])
    VelocityX, VelocityY = 0, 0
    for i in range(len(data)):
        a1 = accel[i, 0]
        a2 = accel[i, 1]
        if i == 347:
            print("0")
        delta_velocityX = -accel[i, 0] * dt
        delta_velocityY = -accel[i, 1] * dt

        delta_angle = -gyro[i] * dt
        angle = angle + delta_angle * 57.2958

        VelocityX = VelocityX + delta_velocityX
        VelocityY = VelocityY + delta_velocityY
        posX = posX + math.cos(angle) * VelocityX * dt
        posY = posY + math.sin(angle) * VelocityY * dt

        result[i, 0] = posX
        result[i, 1] = posY
    return result


def calc_delta_from_accel_gyro(dt, accelX, accelY, gyroZ, real_angle, prev_coordX, prev_coordY, velocityX, velocityY):
    delta_angle = gyroZ * dt
    angle = real_angle + delta_angle

    linear_acceleration = math.sqrt(accelX ** 2 + accelY ** 2)
    delta_velocityX = linear_acceleration * math.sin(angle) * dt
    delta_velocityY = linear_acceleration * math.cos(angle) * dt

    velocityX += delta_velocityX
    velocityY += delta_velocityY
    posX = prev_coordX + velocityX * dt
    posY = prev_coordY + velocityY * dt

    return posX, posY


def visualise_gps(time, gpsX, gpsY, realX, realY):
    plt.subplot(2, 1, 1)
    plt.plot(time, gpsX, color="green", linewidth=3)
    plt.plot(time, realX, color="orange", linewidth=2)
    plt.xlabel("Время, сек")
    plt.ylabel("Положение по Х")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, gpsY, color="green", linewidth=3)
    plt.plot(time, realY, color="orange", linewidth=2)
    plt.xlabel("Время, сек")
    plt.ylabel("Положение по Y")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # Using Qt backend on plots
    matplotlib.use('Qt5Agg')
    # Read file as Pandas series type
    datafile = pd.read_csv("datasets/data_12.0.csv")
    # Read parameters as time, accel, gyro, gps, real_coord
    timestamp = datafile.iloc[:, 0]
    accelX, accelY = datafile.iloc[:, 1], datafile.iloc[:, 2]
    gyroZ = datafile.iloc[:, 6]
    gpsX, gpsY = datafile.iloc[:, 9], datafile.iloc[:, 10]
    real_coordX, real_coordY = datafile[:, 11], datafile.iloc[:, 12]


    visualise_gps(datafile.iloc[:, 0], datafile.iloc[:, 9], datafile.iloc[:, 10], datafile.iloc[:, 11],
                  datafile.iloc[:, 12])
