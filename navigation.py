import math
import numpy as np
import filters


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


def new_calc(data):
    steering_wheel_last = data.iloc[0, 13]
    offset_x = data.iloc[0, 11]
    offset_y = data.iloc[0, 12]
    x0 = 0
    y0 = 0
    dt = 0.05
    result = np.zeros([len(data), 6])
    for i in range(len(data)):
        speed = -data.iloc[i, 8] * 0.277778
        steering_wheel = steering_wheel_last + data.iloc[i, 13] * 1.1
        if i > 20:
            if speed < -0.00:
                delta_velocityX = speed * math.cos(steering_wheel)
                delta_velocityY = speed * math.sin(steering_wheel)
            else:
                delta_velocityX = 0
                delta_velocityY = 0
        else:
            delta_velocityX = 0
            delta_velocityY = 0
        accelX = delta_velocityX / dt
        accelY = delta_velocityY / dt

        x = x0 + delta_velocityX * dt
        y = y0 + delta_velocityY * dt

        x0 = x
        y0 = y

        steering_wheel_last = steering_wheel

        result[i, 0] = accelX
        result[i, 1] = accelY
        result[i, 2] = data.iloc[i, 11]
        result[i, 3] = data.iloc[i, 12]
        result[i, 4] = x + offset_x
        result[i, 5] = y + offset_y

    print(0)