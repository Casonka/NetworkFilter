import math

IS_ADVANCED_CALC = False


def calc_delta_from_accel_gyro(dt, accelX, accelY, gyroZ, real_angle, velocityX, velocityY):
    delta_angle = gyroZ * dt
    angle = real_angle + delta_angle

    linear_acceleration = math.sqrt(accelX ** 2 + accelY ** 2)
    delta_velocityX = linear_acceleration * math.sin(angle) * dt
    delta_velocityY = linear_acceleration * math.cos(angle) * dt

    velocityX += delta_velocityX
    velocityY += delta_velocityY
    posX = velocityX * dt
    posY = velocityY * dt

    return posX, posY

