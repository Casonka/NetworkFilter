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


# [calculating def's]

# prepare_data() - preparing data frames for network
def prepare_data(filepath):
    dataset = pd.read_csv(filepath)
    train_data = dataset.iloc[:, 0:9].values
    target_data = dataset.iloc[:, -6:].values
    return train_data, target_data


# prepare_model() - preparing architecture network (RNN)
def prepare_model(train_data):
    model = keras.Sequential()
    model.add(layers.SimpleRNN(name="rnn_layer_1", units=32, input_shape=(9, 1)))
    model.add(layers.Dense(name="dense_layer_2", units=16, activation="sigmoid"))
    model.add(layers.Dense(name="dense_layer_3", units=6, activation=None))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# --------------------------------------------------------------------------------------------- #

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


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
    # input_train_paths, input_target_data = prepare_data(FILEPATH)
    input_paths, target_paths = prepare_data(FILEPATH)
    input_train_paths = input_paths[0 : 300]
    target_train_paths = target_paths[0 : 300]
    input_val_paths = input_paths[300 :]
    target_val_paths = target_paths[300 :]
    network = prepare_model(input_train_paths)
    network.summary()
    trainGen = BatchGenerator(BATCH_SIZE, DATA_SIZE, input_train_paths, target_train_paths)
    valGen = BatchGenerator(BATCH_SIZE, DATA_SIZE, input_val_paths, target_val_paths)
    network_callback = CustomCallback()
    history = network.fit(trainGen, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=valGen, callbacks=network_callback)
    network.save('model/')
    network.predict()