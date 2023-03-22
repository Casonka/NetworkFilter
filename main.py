# System includes
import os
# ------------------------- #
# TensorFlow includes
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import Sequence
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
TrainData = pd.DataFrame
TrainLabels = pd.DataFrame
# ------------------------- #
# [NO WARNINGS]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------- #
# [Network parameters]
SHUFFLE_BUFFER = 500
BATCH_SIZE = 1
DATA_SIZE = 9
# ------------------------- #
# [EPOCHS]
EPOCHS = 1


# ------------------------- #

# prepare_data() - preparing data frames for network
def prepare_data(filepath):
    dataset = pd.read_csv(filepath)
    train_data = dataset.iloc[:, 0:9]
    target_data = dataset.iloc[:, -6:]
    return train_data, target_data


# prepare_model() - preparing architecture network (RNN)
def prepare_model():
    inputs = keras.Input(shape=(9, 1), dtype="float64")
    x = keras.layers.LSTM(32, activation="sigmoid")(inputs)
    x = keras.layers.Dense(16, activation="sigmoid")(x)
    outputs = keras.layers.Dense(6, activation="sigmoid")(x)
    keras_model = keras.Model(inputs, outputs)
    return keras_model


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
        batch_input_data_paths = self.input_data_paths.iloc[i: i + self.batch_size, :]
        batch_target_data_paths = self.target_data_paths.iloc[i: i + self.batch_size, :]
        return batch_input_data_paths, batch_target_data_paths


if __name__ == '__main__':
    input_train_paths, input_target_data = prepare_data(FILEPATH)
    network = prepare_model()
    trainGen = BatchGenerator(BATCH_SIZE, DATA_SIZE, input_train_paths, input_target_data)
    network.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",
                    metrics=[keras.metrics.SparseCategoricalAccuracy(),
                             keras.metrics.CategoricalCrossentropy(),
                             keras.metrics.Accuracy()])
    history = network.fit(trainGen, epochs=EPOCHS)
