# System includes
import os
from sklearn.utils import shuffle
from random import randint

import filters
import navigation
# ------------------------- #
# TensorFlow includes
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.lite.python.util import convert_bytes_to_c_source
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
BATCH_SIZE = 16
EPOCHS = 700
INPUT_PARAMETERS = len(X_PARAMS)
OUTPUT_PARAMETERS = len(Y_PARAMS)
TIME_INTERVAL_MS = 250  # milliseconds
TIME_TO_VARIABLE = int(TIME_INTERVAL_MS / 50)  # variables to shapes
dt = 0.05
TEST_HEAD = 5
# TEST_HEAD = randint(2, 40)
# [Convert and visualise features]
IS_CONVERT_LITE = True
IS_VISUALISE = False
IS_TRAIN = False
IS_TEST = False
IS_STATISTIC = False
IS_AUGMENTATION = False
IS_SHUFFLE = False
# [Test values]
X_TEST = np.array([[0.206445906, 0.008119201, 0.006391922],
                   [0.205743347, 0.010309473, 0.007049678],
                   [0.207203528, 0.011822678, 0.007799107],
                   [0.207106149, 0.010259509, 0.008076419],
                   [0.207227491, 0.010501682, 0.008305602]])
# 1.05 - 1.40 = 350 milliseconds
Y_TEST = np.array([[-1.58305, 13.89881, 0.20092502, -0.00261093],
                   [-1.58451, 14.61147, 0.21082472, -0.00305459],
                   [-1.58791, 16.03721, 0.23062710, -0.00410544],
                   [-1.58969, 16.75113, 0.24053875, -0.00472469],
                   [-1.59348, 18.18004, 0.26035066, -0.00608674]])


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if IS_TEST:
            print("\r\n" + "[Callback] - Predict " + str(self.model.predict(np.expand_dims(X_TEST, axis=0))))
            print("\r\n" + "[Callback] - Old     " + str(Y_TEST[7, 2:]))


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
# [Defs]
def visualise_training_network(history_train):
    try:
        plt.plot(history_train.history['loss'])
        plt.plot(history_train.history['val_loss'])
        plt.title('Mean squared error')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    except:
        print("Wrong model history")


def visualise_predict(network):
    # read dataframe by pandas
    temp = pd.read_csv(FILEPATH + "dataa_30.0.csv")
    # get input variables
    time = temp[["time"]].drop(labels=[temp.count()[0] - 1]).head(TEST_HEAD).values
    dX = -temp[['true_deltaX']].shift(-1).drop(labels=[temp.count()[0] - 1]).head(TEST_HEAD).values
    y = temp[['true_deltaY']].shift(-1).drop(labels=[temp.count()[0] - 1]).head(TEST_HEAD).values
    y = np.append(dX, y, 1)
    posX = -temp[['truePosX']].drop(labels=[temp.count()[0] - 1]).head(TEST_HEAD).values
    posY = temp[['truePosY']].drop(labels=[temp.count()[0] - 1]).head(TEST_HEAD).values
    accel = temp[['accelX', 'accelY']].drop(labels=[temp.count()[0] - 1]).head(TEST_HEAD).values
    gyro = temp[['gyroZ']].drop(labels=[temp.count()[0] - 1]).head(TEST_HEAD).values

    # filtration without network
    # low pass filter
    # f_accelX = filters.low_pass_filter_calc(accel[:, 0], 0.49)
    # f_accelY = filters.low_pass_filter_calc(accel[:, 1], 0.49)
    # f_gyroZ = filters.low_pass_filter_calc(gyro, 0.49)
    # median filter
    # f_accelX = filters.median_filter_calc(accel[:, 0], TEST_HEAD)
    # f_accelY = filters.median_filter_calc(accel[:, 1], TEST_HEAD)
    # f_gyroZ = filters.median_filter_calc(gyro, TEST_HEAD)
    # moving average filter
    f_accelX = filters.moving_average_filter_calc(accel[:, 0], TEST_HEAD)
    f_accelY = filters.moving_average_filter_calc(accel[:, 1], TEST_HEAD)
    f_gyroZ = filters.moving_average_filter_calc(gyro, TEST_HEAD)
    # alpha beta kalman filter
    # f_accelX = filters.alpha_beta_filter_calc(accel[:, 0])
    # f_accelY = filters.alpha_beta_filter_calc(accel[:, 1])
    # f_gyroZ = filters.alpha_beta_filter_calc(gyro)

    speed = temp[['speed']].drop([temp.count()[0] - 1]).head(TEST_HEAD).values
    speed = speed[speed.shape[0] - 1]
    angle = temp[['compassAngle']].drop([temp.count()[0] - 1]).head(TEST_HEAD).values
    angle = angle[angle.shape[0] - 1]

    dX, dY = navigation.calc_delta_from_accel_gyro(0.05, f_accelX[f_accelX.shape[0] - 1],
                                                    f_accelY[f_accelY.shape[0] - 1],
                                                    f_gyroZ[f_gyroZ.shape[0] - 1], angle, speed, speed)

    accel /= 19.614
    gyro /= 4.36332
    X = np.append(accel, gyro, 1)
    predict = network.predict(np.expand_dims(X, axis=0))

    errorX = (posX[-1] + y[-1, 0]) - (posX[-1] + predict[-1, 0])
    errorY = (posX[-1] + y[-1, 1]) - (posX[-1] + predict[-1, 1])

    f_errorX = (posX[-1] + y[-1, 0]) - (posX[-1] + dX)
    f_errorY = (posX[-1] + y[-1, 1]) - (posX[-1] + dY)
    print("filter predict error " + str(f_errorX) + "  " + str(f_errorY))
    print("Network filter predict error " + str(errorX) + "  " + str(errorY))
    # по оси X
    plt.subplot(2, 1, 1)
    # Рисуем перемещение до нужного момента
    plt.plot(time[:-1], posX[:-1], color='blue')
    # Отмечаем реальную точку
    plt.plot(time[-1], (posX[-1] + y[-1, 0]), color='green', marker='D')
    # Отмечаем предсказанную точку
    plt.plot(time[-1], (posX[-1] + predict[-1, 0]), color='red', marker='x')
    plt.plot(time[-1], (posX[-1] + dX), color='purple', marker='x')
    plt.xlabel("Время, с")
    plt.ylabel("Координата X автомобиля, м")
    # по оси Y
    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], posY[:-1], color='blue')
    plt.plot(time[-1], (posY[-1] + y[-1, 1]), color='green', marker='D')
    plt.plot(time[-1], (posY[-1] + predict[-1, 1]), color='red', marker='x')
    plt.plot(time[-1], (posY[-1] + dY), color='purple', marker='x')
    plt.xlabel("Время, с")
    plt.ylabel("Координата Y автомобиля, м")
    plt.show()


# Preparing Dataframes -> numpy array for train
def prepare_dataset(filepath):
    tempX_data = np.zeros([0, INPUT_PARAMETERS])
    tempY_data = np.zeros([0, OUTPUT_PARAMETERS])
    for i, file in enumerate(os.listdir(FILEPATH)):
        if file.endswith(".csv"):
            temp = pd.read_csv(FILEPATH + file)
            dX = -temp[['true_deltaX']].shift(-1).drop(labels=[temp.count()[0] - 1]).values
            y = temp[['true_deltaY']].shift(-1).drop(labels=[temp.count()[0] - 1]).values
            y = np.append(dX, y, 1)
            accel = temp[['accelX', 'accelY']].drop(labels=[temp.count()[0] - 1]).values / 19.614
            gyro = temp[['gyroZ']].drop(labels=[temp.count()[0] - 1]).values / 4.36332
            # compass = temp[['compassAngle']].drop(labels=[temp.count()[0] - 1]).values / 3.14157
            # speed = temp[['speed']].drop(labels=[temp.count()[0] - 1]).values / 60.0
            x = np.append(accel, gyro, 1)
            # x = np.append(x, compass, 1)
            # x = np.append(x, speed, 1)
            tempY_data = np.append(tempY_data, y, 0)
            tempX_data = np.append(tempX_data, x, 0)
            if IS_AUGMENTATION:
                tempY_data = np.append(tempY_data, -y, 0)
                tempX_data = np.append(tempX_data, -x, 0)

    size = tempX_data.shape[0]
    if IS_SHUFFLE:
        tempX_data, tempY_data = shuffle(tempX_data, tempY_data)
    if IS_STATISTIC:
        print("[Dataset size] - " + str(size))
        print("[Time map] - " + str(size * 50 / 1000) + " seconds")
    return tempX_data[:round(size * 0.7)], tempX_data[round(size * 0.7):], \
        tempY_data[:round(size * 0.7)], tempY_data[round(size * 0.7):]


# Preparing model architecture (RNN)
def get_model():
    try:
        tmp_model = keras.models.load_model("model/model.h5", compile=False)
        tmp_model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse", metrics="mae")
    except IOError:
        print("No such h5 model file, creating new model")
        tmp_model = keras.Sequential(name="model")
        tmp_model.add(layers.InputLayer(batch_input_shape=(None, TIME_TO_VARIABLE, INPUT_PARAMETERS), name="input_1"))
        tmp_model.add(layers.SimpleRNN(8, name="rnn_2", return_sequences=True))
        tmp_model.add(layers.Dropout(0.3))
        tmp_model.add(layers.SimpleRNN(6, name="rnn_3",))
        tmp_model.add(layers.Dense(2, name="dense_4",))
        tmp_model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse", metrics="mae")
        if IS_STATISTIC:
            tmp_model.summary()
    return tmp_model


# Convert the model to tflite format and C (.c .h) array
def get_lite_model():
    try:
        tmp_model = keras.models.load_model("model/model.h5")

        converter = tf.lite.TFLiteConverter.from_keras_model(tmp_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()

        # Save the model.
        with open('model/model.tflite', 'wb') as f:
            f.write(tflite_model)

        source_text, header_text = convert_bytes_to_c_source(tflite_model, "model", include_path="model/lite/")

        with open("model/lite" + '.h', 'w') as file:
            file.write(header_text)
        with open("model/lite" + '.cc', 'w') as file:
            file.write(source_text)
    except IOError:
        print("No such h5 model file")
    finally:
        accel = pd.read_csv("datasets/dataa_30.0.csv")[['accelX', 'accelY']].head(50).values / 19.614
        gyroZ = pd.read_csv("datasets/dataa_30.0.csv")[['gyroZ']].head(50).values / 4.36332
        true_delta = pd.read_csv("datasets/dataa_30.0.csv")[['true_deltaX', 'true_deltaX']].head(50).values

        out = np.append(accel, gyroZ, 1)
        # out = np.expand_dims(out, axis=0)
        np.savetxt("validation_x.csv", out, delimiter=',')
        np.savetxt("validation_y.csv", true_delta, delimiter=',')


# --------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    # Preparing train and validation data
    # --------------------------------#
    train_valX_paths, validation_valX_paths, train_valY_paths, \
        validation_valY_paths = prepare_dataset(FILEPATH)

    # clear old gpu/cpu allocated memory
    tf.keras.backend.clear_session()
    # Prepare and compile model
    # --------------------------------#
    model = get_model()

    # Prepare train and validation batch generators
    # --------------------------------#
    if IS_TRAIN:
        trainGen = BatchGenerator(train_valX_paths, train_valY_paths)
        valGen = BatchGenerator(validation_valX_paths, validation_valY_paths)

        epoch_callback = CustomCallback()
        # save model in training
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="model/model.h5",
                                                           save_best_only=True,
                                                           monitor="val_loss", mode="min", verbose=1)
        # stop training
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")

        csv_logger = keras.callbacks.CSVLogger(filename='model/train.log',
                                               append=False, separator=',')

        callbacks = [epoch_callback, early_stopping, model_checkpoint, csv_logger]

        history = model.fit(trainGen, epochs=EPOCHS, validation_data=valGen, callbacks=callbacks, shuffle=IS_SHUFFLE)

        # visualise training
        if IS_VISUALISE:
            visualise_training_network(history)
            visualise_predict(model)

    # creating tflite model and C files
    if IS_CONVERT_LITE:
        get_lite_model()
