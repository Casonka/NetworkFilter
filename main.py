# System includes
import os
from abc import ABC
import navigation
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
FILEPATH = "datasets/"
# ------------------------- #
# [Pandas variables]
X_PARAMS = ['accelX', 'accelY', 'gyroZ']
Y_PARAMS = ['true_deltaX', 'true_deltaY']
# ------------------------- #
# [NO WARNINGS]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------- #
# [Network parameters]
BATCH_SIZE = 25
INPUT_PARAMETERS = 3
OUTPUT_PARAMETERS = 2
TIME_INTERVAL_MS = 200  # milliseconds
TIME_TO_VARIABLE = int(TIME_INTERVAL_MS / 50)  # variables to shapes
EPOCHS = 1


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())


# Batch Generator Class
class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_data_paths, target_data_paths):
        self.input_data_paths = input_data_paths
        self.target_data_paths = target_data_paths

    def __len__(self):
        return len(self.target_data_paths // TIME_TO_VARIABLE) // BATCH_SIZE

    def __getitem__(self, idx):
        x = np.zeros((0, TIME_TO_VARIABLE, INPUT_PARAMETERS))
        y = np.zeros((0, OUTPUT_PARAMETERS))
        for i in range(BATCH_SIZE):
            index = i * TIME_TO_VARIABLE
            temp = self.input_data_paths[index: index + TIME_TO_VARIABLE]
            temp = np.expand_dims(temp, 0)
            x = np.append(x, temp, 0)

            temp = self.target_data_paths[index + TIME_TO_VARIABLE - 1]
            temp = np.expand_dims(temp, 0)
            y = np.append(y, temp, 0)
        return x, y


# ------------------------- #
# [calculating def's]
# prepare_data() - preparing data frames for network
def prepare_dataset(filepath):
    tempX_data = np.zeros([0, INPUT_PARAMETERS])
    tempY_data = np.zeros([0, OUTPUT_PARAMETERS])
    for i, file in enumerate(os.listdir(FILEPATH)):
        if file.endswith(".csv"):
            temp = pd.read_csv(FILEPATH + file)
            y = temp[Y_PARAMS].shift(-1).drop(labels=[temp.count()[0] - 1]).values
            x = temp[X_PARAMS].drop(labels=[temp.count()[0] - 1]).values
            tempY_data = np.append(tempY_data, y, 0)
            tempX_data = np.append(tempX_data, x, 0)

    size = tempX_data.shape[0]

    return tempX_data[:round(size * 0.7)], tempX_data[round(size * 0.7):], \
        tempY_data[:round(size * 0.7)], tempY_data[round(size * 0.7):]


# prepare_model() - preparing architecture network (RNN)
def get_model():
    try:
        tmp_model = keras.models.load_model("model/model.h5")
    except IOError:
        print("No such h5 model file, creating new model")
        tmp_model = keras.Sequential(name="model")
        tmp_model.add(layers.InputLayer(batch_input_shape=(None, None, INPUT_PARAMETERS), name="input_1"))
        tmp_model.add(layers.Dense(6, activation="tanh", name="dense_2"))
        tmp_model.add(layers.LSTM(12, name="lstm_3"))
        tmp_model.add(layers.Dense(2, name="dense_4"))
        tmp_model.compile(optimizer=keras.optimizers.Adam(0.001), loss=["mse"], metrics=["mse", "mae"])
    return tmp_model


# --------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    # Preparing train and validation data
    # --------------------------------#
    train_valX_paths, validation_valX_paths, train_valY_paths, \
        validation_valY_paths = prepare_dataset(FILEPATH)

    # Prepare and compile model
    # --------------------------------#
    model = get_model()
    model.summary()

    # Prepare train and validation batch generators
    # --------------------------------#
    trainGen = BatchGenerator(train_valX_paths, train_valY_paths)
    valGen = BatchGenerator(validation_valX_paths, validation_valY_paths)

    epoch_callback = CustomCallback()
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="model/model.h5", save_best_only=True,
                                                       monitor="val_loss", mode="min")
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min")

    callbacks = [epoch_callback, model_checkpoint, early_stopping]

    history = model.fit(trainGen, epochs=EPOCHS, validation_data=valGen, callbacks=callbacks)

    # Convert the model to tflite format.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    # Save the model.
    with open('model/model.tflite', 'wb') as f:
        f.write(tflite_model)

    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model mean squared error')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


