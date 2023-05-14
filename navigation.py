import numpy as np
import filters


def integrating_calc(accelerometer, gyroscope):
    pos_x, pos_y = np.zeros(filters.ITEMS)
    for i in filters.ITEMS:
        speed = accelerometer[i] * 0.05
