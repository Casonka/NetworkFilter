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
# Filter defs


def alpha_beta_filter_calc(data):
    k_kalman = 0
    T_0 = 0.05
    filtered_velocity = 0.0
    filtered_value = np.zeros((len(data)))
    extrapolated_value = 0.
    extrapolated_velocity = 0.
    for i in range(len(data)):
        k_kalman += 1
        if k_kalman == 1:
            filtered_value[i] = data[i]
            continue
        if k_kalman == 2:
            filtered_velocity = data[i] - filtered_value[i - 1] / T_0
            filtered_value[i] = data[i]

            extrapolated_value = filtered_value[i] + (filtered_velocity * T_0)
            extrapolated_velocity = filtered_velocity
            continue

        alpha = (2.0 * (2.0 * k_kalman - 1.0)) / (k_kalman * (k_kalman + 1.0))
        beta = 6.0 / (k_kalman * (k_kalman + 1))

        filtered_value[i] = extrapolated_value + (alpha * (data[i] - extrapolated_value))
        filtered_velocity = extrapolated_velocity + (beta / T_0 * (data[i] - extrapolated_value))

        filtered_value[i] = filtered_value[i] + (filtered_velocity * T_0)
        extrapolated_velocity = filtered_velocity
    return filtered_value


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
        tmp = np.sort(tmp)
        if len(tmp) < 2:
            filtered[i] = tmp
        else:
            filtered[i] = tmp[tmp.shape[0] // 2]
    return filtered

