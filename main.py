# System includes
import os
from sklearn.utils import shuffle
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
X_PARAMS = ['accelX', 'accelY', 'gyroZ', 'compassAngle', 'speed']
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
TIME_INTERVAL_MS = 600  # milliseconds
TIME_TO_VARIABLE = int(TIME_INTERVAL_MS / 50)  # variables to shapes
dt = 0.05
# [Convert and visualise features]
IS_CONVERT_LITE = False
IS_VISUALISE = True
IS_TRAIN = True
IS_TEST = True
IS_STATISTIC = False
IS_AUGMENTATION = False
IS_SHUFFLE = False
# [Test values]
X_TEST = np.array([[0.206445906, 0.008119201, 0.006391922, -0.503904099, 0.231646833],
                   [0.205743347, 0.010309473, 0.007049678, -0.504368835, 0.2435245],
                   [0.207203528, 0.011822678, 0.007799107, -0.504368835, 0.267286833],
                   [0.207106149, 0.010259509, 0.008076419, -0.505451096, 0.2791855],
                   [0.207227491, 0.010501682, 0.008305602, -0.506017692, 0.303000667],
                   [0.208451616, 0.011094116, 0.008608124, -0.507224095, 0.314905],
                   [0.206712042, 0.013711635, 0.009206292, -0.507879818, 0.3268125],
                   [0.207784236, 0.011007444, 0.009169623, -0.50850371, 0.338723333]])
# 1.05 - 1.40 = 350 milliseconds
Y_TEST = np.array([[-1.58305, 13.89881, -0.20092502, -0.00261093],
                   [-1.58451, 14.61147, -0.21082472, -0.00305459],
                   [-1.58791, 16.03721, -0.23062710, -0.00410544],
                   [-1.58969, 16.75113, -0.24053875, -0.00472469],
                   [-1.59348, 18.18004, -0.26035066, -0.00608674],
                   [-1.59554, 18.89430, -0.27025340, -0.00682930],
                   [-1.59750, 19.60875, -0.28016359, -0.00766041],
                   [-1.59951, 20.32340, -0.29006710, -0.00845993]])


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if IS_TEST:
            print("\r\n" + "[Callback] - Predict " + str(self.model.predict(np.expand_dims(X_TEST, axis=0))))
            print("\r\n" + "[Callback] - Old     " + str(Y_TEST[7, 2:]))

    # def on_predict_end(self, logs=None):
    #     if IS_TEST:


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


def custom_loss_function(y_true, y_pred):
    delta_pred2d = tf.zeros([0, 2])
    accelX = tf.reshape(y_pred[:, 0], (-1, 1)) * 19.614
    accelY = tf.reshape(y_pred[:, 1], (-1, 1)) * 19.614
    gyroZ = tf.reshape(y_pred[:, 2], (-1, 1)) * 4.36332
    compass_angle = tf.reshape(y_true[:, 0], (-1, 1))
    speed = tf.reshape(y_true[:, 1], (-1, 1))
    # calc new angle
    angle = compass_angle + (gyroZ * dt)
    # calc acceleration vector
    linear_acceleration = tf.sqrt(accelX ** 2 + accelY ** 2)
    # calc delta velocity X,Y
    delta_velocityX = linear_acceleration * tf.sin(angle) * dt
    delta_velocityY = linear_acceleration * tf.cos(angle) * dt
    velocityX = speed + delta_velocityX
    velocityY = speed + delta_velocityY
    # calc delta X,Y
    pred_coordX = velocityX * dt
    pred_coordY = velocityY * dt
    array = tf.keras.backend.concatenate((pred_coordX, pred_coordY), 1)
    delta_pred2d = tf.keras.backend.concatenate((delta_pred2d, array), 0)
    delta_true = y_true[:, 2:]
    result = tf.reduce_mean(tf.abs(delta_true - delta_pred2d), 0)
    return result


# ------------------------- #
# [Defs]
def visualise_training_network(history_train):
    try:
        plt.plot(history_train.history['loss'])
        plt.plot(history_train.history['val_loss'])
        plt.title('Amplitude abs error')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    except:
        print("Wrong model history")


# Preparing Dataframes -> numpy array for train
def prepare_dataset(filepath):
    tempX_data = np.zeros([0, INPUT_PARAMETERS])
    tempY_data = np.zeros([0, OUTPUT_PARAMETERS])
    for i, file in enumerate(os.listdir(FILEPATH)):
        if file.endswith(".csv"):
            temp = pd.read_csv(FILEPATH + file)
            y = temp[Y_PARAMS].shift(-1).drop(labels=[temp.count()[0] - 1]).values
            accel = temp[['accelX', 'accelY']].drop(labels=[temp.count()[0] - 1]).values / 19.614
            gyro = temp[['gyroZ']].drop(labels=[temp.count()[0] - 1]).values / 4.36332
            compass = temp[['compassAngle']].drop(labels=[temp.count()[0] - 1]).values / 3.14157
            speed = temp[['speed']].drop(labels=[temp.count()[0] - 1]).values / 60.00
            x = np.append(accel, gyro, 1)
            x = np.append(x, compass, 1)
            x = np.append(x, speed, 1)
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
        tmp_model.compile(optimizer=keras.optimizers.Adam(0.0001), loss="mse", metrics="mae")
    except IOError:
        print("No such h5 model file, creating new model")
        tmp_model = keras.Sequential(name="model")
        tmp_model.add(layers.InputLayer(batch_input_shape=(None, None, INPUT_PARAMETERS), name="input_1"))
        tmp_model.add(layers.SimpleRNN(9, name="rnn_2", return_sequences=True))
        tmp_model.add(layers.Dropout(0.3))
        tmp_model.add(layers.SimpleRNN(6, name="rnn_3",))
        tmp_model.add(layers.Dense(2, name="dense_4",))
        tmp_model.compile(optimizer=keras.optimizers.Adam(0.0001), loss="mse", metrics="mae")
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

        source_text, header_text = convert_bytes_to_c_source(tflite_model, "lite", include_path="model/lite/")

        with open("lite" + '.h', 'w') as file:
            file.write(header_text)
        with open("lite" + '.cc', 'w') as file:
            file.write(source_text)
    except IOError:
        print("No such h5 model file")


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
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath="model/model.h5", save_best_only=True,
                                                           monitor="val_loss", mode="min")
        # stop training
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min")

        csv_logger = keras.callbacks.CSVLogger(filename='train_log.log', append=False, separator=',')

        callbacks = [epoch_callback, early_stopping, model_checkpoint, csv_logger]

        history = model.fit(trainGen, epochs=EPOCHS, validation_data=valGen, callbacks=callbacks, shuffle=IS_SHUFFLE)

        # creating tflite model and C files
        if IS_CONVERT_LITE:
            get_lite_model()

        # visualise training
        if IS_VISUALISE:
            visualise_training_network(history)
