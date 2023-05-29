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
X_PARAMS = ['accelX', 'accelY', 'gyroZ']
Y_PARAMS = ['compassAngle', 'speed', 'true_deltaX', 'true_deltaY']
# ------------------------- #
# [NO WARNINGS]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------- #
# [Network parameters]
BATCH_SIZE = 15
EPOCHS = 100
INPUT_PARAMETERS = len(X_PARAMS)
OUTPUT_PARAMETERS = len(Y_PARAMS)
TIME_INTERVAL_MS = 500  # milliseconds
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
X_TEST = np.array([[4.04923, 0.15925, 0.02789],
                   [4.03545, 0.20221, 0.03076],
                   [4.06409, 0.23189, 0.03403],
                   [4.06218, 0.20123, 0.03524],
                   [4.06456, 0.20598, 0.03624],
                   [4.08857, 0.21760, 0.03756],
                   [4.05445, 0.26894, 0.04017],
                   [4.07548, 0.21590, 0.04001]])
# 1.05 - 1.40 = 350 milliseconds
Y_TEST = np.array([[-1.58305, 13.89881, -0.20092502, -0.00261093],
                   [-1.58615, 15.32418, -0.21082472, -0.00305459],
                   [-1.58451, 14.61147, -0.21082472, -0.00305459],
                   [4.06218, 0.20123, 0.03524],
                   [4.06456, 0.20598, 0.03624],
                   [4.08857, 0.21760, 0.03756],
                   [4.05445, 0.26894, 0.04017],
                   [4.07548, 0.21590, 0.04001]])


1.15000,,9.72705,0.00150,-0.00038,,,2.80691,-4.71288,-2.56568,-0.02110,-0.02,-0.22072240,-0.00355293
1.20000,,9.80084,0.00066,-0.00042,-1.58791,16.03721,1.41535,-1.03261,-2.79630,-0.02521,-0.03,-0.23062710,-0.00410544
1.25000,,9.84038,0.00047,-0.00032,-1.58969,16.75113,-0.22428,-0.13040,-3.03684,-0.02993,-0.02,-0.24053875,-0.00472469
1.30000,,9.74503,0.00056,-0.00014,-1.59155,17.46560,-3.06207,-1.16917,-3.28729,-0.03530,-0.03,-0.25044652,-0.00536735
1.35000,,9.77922,0.00146,0.00005,-1.59348,18.18004,-3.12659,-2.74779,-3.54764,-0.04139,-0.02,-0.26035066,-0.00608674
1.40000,,9.80600,0.00045,0.00013,-1.59554,18.89430,-3.60735,-3.16249,-3.81789,-0.04822,-0.02,-0.27025340,-0.00682930

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if IS_TEST:
            print("\r\n" + "[Callback] - Predict " + str(self.model.predict(np.expand_dims(X_TEST, axis=0))))

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
    compass_angle = tf.abs(tf.reshape(y_true[:, 0], (-1, 1)))
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
        plt.ylabel('AE')
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
            if not file.endswith("-12.0.csv"):
                y = y * -1
            accel = temp[['accelX', 'accelY']].drop(labels=[temp.count()[0] - 1]).values / 19.614
            gyro = temp[['gyroZ']].drop(labels=[temp.count()[0] - 1]).values / 4.36332
            x = np.append(accel, gyro, 1)
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
        tmp_model.compile(optimizer=keras.optimizers.Adam(0.0001), loss=custom_loss_function)
    except IOError:
        print("No such h5 model file, creating new model")
        tmp_model = keras.Sequential(name="model")
        tmp_model.add(layers.InputLayer(batch_input_shape=(None, None, INPUT_PARAMETERS), name="input_1"))
        tmp_model.add(layers.Dense(6, activation="tanh", name="dense_2"))
        tmp_model.add(layers.LSTM(12, name="rnn_3", return_sequences=True))
        tmp_model.add(layers.Dropout(0.5))
        tmp_model.add(layers.LSTM(6, name="rnn_4", return_sequences=False))
        tmp_model.add(layers.Dense(3, name="dense_5", activation="tanh"))
        tmp_model.compile(optimizer=keras.optimizers.Adam(0.0001), loss=custom_loss_function)
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
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, mode="min")

        callbacks = [epoch_callback, model_checkpoint, early_stopping]

        history = model.fit(trainGen, epochs=EPOCHS, validation_data=valGen, callbacks=callbacks)

        # creating tflite model and C files
        if IS_CONVERT_LITE:
            get_lite_model()

        # visualise training
        if IS_VISUALISE:
            visualise_training_network(history)
