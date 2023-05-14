# System includes
import os
from abc import ABC

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
# FILEPATH = "datasets/data_20.0.csv"
FILEPATH = "datasets/data_30.0.csv"
# FILEPATH = "data_40.0.csv"
# FILEPATH = "data_50.0.csv"
# ------------------------- #
# [Pandas variables]

# ------------------------- #
# [NO WARNINGS]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------- #
# [Network parameters]
BATCH_SIZE = 1
DATA_SIZE = 6
# ------------------------- #
# [EPOCHS]
EPOCHS = 1


class CustomCallback(tf.keras.callbacks.Callback):

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())

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


# ------------------------- #
# [calculating def's]

# prepare_data() - preparing data frames for network
def prepare_data(filepath):
    dataset = pd.read_csv(filepath)
    size = len(dataset.index)

    tmp = dataset.iloc[:, 1:7].values

    trainX_data = tmp[: round(size * 0.7)]
    validationX_data = tmp[round(size * 0.7):]

    validationY_data = dataset.iloc[:, -6:].values[: round(size * 0.7)]
    trainY_data = dataset.iloc[:, -6:].values[round(size * 0.7):]
    return trainX_data, validationX_data, trainY_data, validationY_data


# prepare_model() - preparing architecture network (RNN)
def prepare_model():
    tmp_model = keras.Sequential()
    tmp_model.add(layers.SimpleRNN(batch_size=BATCH_SIZE, name="rnn_layer_1", units=32, input_shape=(6, 1)))
    tmp_model.add(layers.Dense(batch_size=BATCH_SIZE, name="dense_layer_2", units=16, activation="sigmoid"))
    tmp_model.add(layers.Dense(batch_size=BATCH_SIZE, name="dense_layer_3", units=6, activation=None))
    tmp_model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['mse'])
    return tmp_model


# --------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    # Preparing train and validation data
    # --------------------------------#
    train_valX_paths, validation_valX_paths, train_valY_paths, \
        validation_valY_paths = prepare_data(FILEPATH)

    # Prepare and compile model
    # --------------------------------#
    model = prepare_model()
    model.summary()

    # Prepare train and validation batch generators
    # --------------------------------#
    trainGen = BatchGenerator(BATCH_SIZE, DATA_SIZE, train_valX_paths, train_valY_paths)
    valGen = BatchGenerator(BATCH_SIZE, DATA_SIZE, validation_valX_paths, validation_valY_paths)

    epoch_callback = CustomCallback()
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='model/my_model.h5', save_best_only=True,
                                                                verbose=1, monitor='val_mse')

    history = model.fit(trainGen, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=valGen,
                        callbacks=[epoch_callback, model_checkpoint_callback])

    check_dataset = pd.read_csv("datasets/data_20.0.csv")
    time = check_dataset.iloc[350:400, 0].values
    check_data = check_dataset.iloc[350:400, 1].values
    buffer = np.zeros([50, 6])

    for i in range(50):
        test_data = check_dataset.iloc[350+i, 1:7]
        predict = model.predict(test_data)
        buffer[i, :] = predict

    plt.plot(time, check_data, color="blue")
    plt.plot(time, buffer[:, 1], color="orange")
