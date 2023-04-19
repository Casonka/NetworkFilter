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
# [Pandas variables]

# ------------------------- #
# [NO WARNINGS]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------- #
# [Network parameters]
BATCH_SIZE = 20
DATA_SIZE = 9
# ------------------------- #
# [EPOCHS]
EPOCHS = 2


# ------------------------- #


# prepare_data() - preparing data frames for network
def prepare_data(filepath):
    dataset = pd.read_csv(filepath)
    train_data = dataset.iloc[:, 0:9].values
    target_data = dataset.iloc[:, -6:].values
    return train_data, target_data


# prepare_model() - preparing architecture network (RNN)
def prepare_model(train_data):
    model = keras.Sequential()
    model.add(layers.SimpleRNN(units=32, input_shape=(9, 1)))
    model.add(layers.Dense(units=16, activation="sigmoid"))
    model.add(layers.Dense(units=6, activation=None))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def filter_low_freq_show(filepath):
    file = pd.read_csv(filepath)


# --------------------------------------------------------------------------------------------- #


# Batch Generator Class
class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch, data_size, input_data_paths, target_data_paths):
        self.batch_size = batch
        self.data_size = data_size
        self.input_data_paths = input_data_paths
        self.target_data_paths = target_data_paths

    def __len__(self):
        return len(self.target_data_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        x = self.input_data_paths[i: i + self.batch_size, :]
        y = self.target_data_paths[i: i + self.batch_size, :]
        return x, y


if __name__ == '__main__':
    input_train_paths, input_target_data = prepare_data(FILEPATH)
    network = prepare_model(input_train_paths)
    trainGen = BatchGenerator(BATCH_SIZE, DATA_SIZE, input_train_paths, input_target_data)
    history = network.fit(trainGen, epochs=EPOCHS, batch_size=BATCH_SIZE)
