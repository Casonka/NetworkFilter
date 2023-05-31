# импортируем библиотеки
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt, transforms

import seaborn as sns
from matplotlib.patches import Ellipse

sns.set()

import random

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# инициируем таблицу ошбок среднеквадратичных отклонений
table_errors_test = pd.DataFrame(index = ['MSE_test'])

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_correlated_dataset(n, dependency, mu, scale):
    latent = np.random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]

def draw_graphV1(dataframe):
    sns_plot = sns.pairplot(dataframe)
    sns_plot.fig.suptitle('График №1 "Парные отношения признаков и истинных ответов"',
                          y=1.03, fontsize=14, fontweight='bold')
    plt.show()


def draw_graph2(dataframe):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
    dependency_nstd = [[0.8, 0.75],
                       [-0.2, 0.35]]
    fig.suptitle('График №2 "Отношение всех параметров к перемещению"', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.85)
    ax1.scatter(dataframe['gpsY'], dataframe['dX'], color='orange')
    ax1.scatter(dataframe['gpsX'], dataframe['dX'], color='red')
    ax1.scatter(dataframe['speed'], dataframe['dX'], color='blue')
    ax1.scatter(dataframe['compass'], dataframe['dX'], color='purple')
    ax1.scatter(dataframe['gyroZ'], dataframe['dX'], color='brown')
    ax1.scatter(dataframe['gyroY'], dataframe['dX'], color='grey')
    ax1.scatter(dataframe['gyroX'], dataframe['dX'], color='gold')
    ax1.scatter(dataframe['accelZ'], dataframe['dX'], color='blue')
    ax1.scatter(dataframe['accelY'], dataframe['dX'], color='pink')
    ax1.scatter(dataframe['accelX'], dataframe['dX'], color='green')
    mu = 100, 815
    scale = 12, 25
    x, y = get_correlated_dataset(1000, dependency_nstd, mu, scale)
    confidence_ellipse(x, y, ax1,
                       alpha=0.5, facecolor='pink', edgecolor='red', zorder=0)
    ax1.set_title('Зависимость признаков от перемещения')
    ax1.set_xlabel('Нормированный ряд признаков, -1/1')
    ax1.set_ylabel('Перемещение, м')

    #fig.savefig('graph_2.png')
    plt.show()


# напишем функцию определения среднеквадратичной ошибки
def error(x_train, x_test, y_train, y_test):
    # инициируем модель линейной регрессии
    model = LinearRegression()
    # обучим модель на обучающей выборке
    model_fit = model.fit(x_train,y_train)
    # сформируем вектор прогнозных значений
    y_pred = model_fit.predict(x_test)
    # определим среднеквадратичную шибку
    error = round(mean_squared_error(y_test, y_pred),3)
    return error


if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    np.random.seed(0)
    N = 45000
    mu = np.array((100., 100, 100, 100, 100, 100, 100, 100, 100, 100))
    accel = pd.read_csv("datasets/dataa_30.0.csv")[['accelX', 'accelY', 'accelZ']].head(N).values / 19.614
    gyro = pd.read_csv("datasets/dataa_30.0.csv")[['gyroX', 'gyroY', 'gyroZ']].head(N).values / 4.36332 * 57.2958
    speed = pd.read_csv("datasets/dataa_30.0.csv")[['speed']].head(N).values / 60.
    compass = pd.read_csv("datasets/dataa_30.0.csv")[['compassAngle']].head(N).values / 3.14157
    gps = pd.read_csv("datasets/dataa_30.0.csv")[['gpsX', 'gpsY']].head(N).values / 120.
    data = np.hstack((accel, gyro))
    data = np.hstack((data, compass))
    data = np.hstack((data, speed))
    data = np.hstack((data, gps))

    cov = np.cov(data.reshape((10, N)))

    X12 = np.dot(data, cov) + mu
    x1 = X12[:, 0]
    x2 = X12[:, 1]
    x3 = X12[:, 2]
    x4 = X12[:, 3]
    x5 = X12[:, 4]
    x6 = X12[:, 5]
    x7 = X12[:, 6]
    x8 = X12[:, 7]
    x9 = X12[:, 8]
    x10 = X12[:, 9]

    X = np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((
        np.vstack((np.vstack((np.vstack((x1, x2)), x3)), x4)), x5)), x6)), x7)), x8)), x9)), x10)).T

    r_0 = 0
    R = np.array((1.3, 1.3, 0.8, 0.3, 0.3, 1.0, 0.8, 0.8, 0.8, 0.8))
    R = R.reshape(-1, R.shape[0])
    e = np.array([random.uniform(-1., 1.) for i in range(N)]).reshape(-1, 1)
    y = r_0 + np.dot(X, R.T) + e
    columns_x = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 'compass', 'speed', 'gpsX', 'gpsY']
    dataframe = pd.DataFrame(np.hstack((X, y)))
    dataframe.columns = columns_x + ['dX']

    #draw_graphV1(dataframe)
    draw_graph2(dataframe)
    #разделим выборку на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # проведем центрирование данных (функция нормирования отключена)
    scaler = StandardScaler(with_mean = True, with_std = False)
    scaler = scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # инициируем модель PCA с 8 компонентами
    model_pca = PCA(n_components=10)
    # обучим модель на обучающей выборке
    model_pca.fit(X_train_norm)
    # преобразуем данные обучающей выборки
    Z_train_norm = model_pca.transform(X_train_norm)
    # преобразуем данные тестовой выборки
    Z_test_norm = model_pca.transform(X_test_norm)

    argument_1X = np.hstack((X_train_norm[:, :2], X_train_norm[:, 5:6]))
    argument_2X = X_train_norm[:, :7]
    argument_3X = np.hstack((X_train_norm[:, :7], X_train_norm[:, 8:9]))

    argument_1Y = np.hstack((X_test_norm[:, :2], X_test_norm[:, 5:6]))
    argument_2Y = X_test_norm[:, :7]
    argument_3Y = np.hstack((X_test_norm[:, :7], X_test_norm[:, 8:9]))

    Z_1X = np.hstack((Z_train_norm[:, :2], Z_train_norm[:, 5:6]))
    Z_2X = Z_train_norm[:, :7]
    Z_3X = np.hstack((Z_train_norm[:, :7], Z_train_norm[:, 8:9]))

    Z_1Y = np.hstack((Z_test_norm[:, :2], Z_test_norm[:, 5:6]))
    Z_2Y = Z_test_norm[:, :7]
    Z_3Y = np.hstack((Z_test_norm[:, :7], Z_test_norm[:, 8:9]))

    # сформируем в pandas таблицу оценок качества модели линейной регрессии в зависимости от используемых признаков
    table_errors_test['All'] = error(X_train_norm, X_test_norm, y_train, y_test)
    table_errors_test['accelX, accelY, gyroZ'] = error(argument_1X, argument_1Y, y_train, y_test)
    table_errors_test['accelX, accelY, gyroZ + speed'] = error(argument_3X, argument_3Y, y_train, y_test)
    table_errors_test['accelX, accelY, gyroZ + speed + compass'] = error(argument_3X, argument_3Y, y_train, y_test)
    table_errors_test['accel + gyro'] = error(argument_2X, argument_2Y, y_train, y_test)
    table_errors_test['accel + gyro + speed'] = error(argument_3X, argument_3Y, y_train, y_test)

    table_errors_test['Все компоненты'] = error(Z_train_norm, Z_test_norm, y_train, y_test)
    table_errors_test['1,2,6 компоненты'] = error(Z_1X, Z_1Y, y_train, y_test)
    table_errors_test['1 - 6 компоненты'] = error(Z_2X, Z_2Y, y_train, y_test)
    table_errors_test['1 - 7 компоненты'] = error(Z_3X, Z_3Y, y_train, y_test)



    print ('Таблица №1 "Сравнение качества модели линейной регрессии, обученной на различных признаках"')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(table_errors_test)

    # формируем таблицу основных описательных статистик исходного пространства
    X_train_dataframe = pd.DataFrame(X_train_norm)
    X_train_dataframe.columns = columns_x
    X_df_describe = X_train_dataframe.describe(percentiles = []).round(3)

    # формируем матрицу преобразованного пространства
    Z_train_norm = model_pca.transform(X_train_norm)
    Z_train_dataframe = pd.DataFrame(Z_train_norm)
    columns_z = ['1-я компонента','2-я компонента', '3-я компонента','4-я компонента', '5-я компонента', '6-я компонента',
                 '7-я компонента', '8-я компонента', '9-я компонента', '10-я компонента']
    Z_train_dataframe.columns = columns_z

    # формируем таблицу основных описательных статистик исходного пространства
    Z_df_describe = Z_train_dataframe.describe(percentiles = []).round(3)

    # формируем сравнительную таблицу основных описательных статистик двух признаковых пространств
    df_describe = pd.concat((X_df_describe, Z_df_describe), axis = 1)
    columns_xz = columns_x + columns_z

    print ('Таблица №2 "Сравнение значений основных описательных статистик исходного и преобразованного пространства"')
    print(df_describe[columns_xz])

    # формируем матрицу ковариации исходных признаков
    X_df_cov = X_train_dataframe.cov().round(3)

    # формируем матрицу ковариации преобразованных признаков
    Z_df_cov = Z_train_dataframe.cov().round(3)

    # формируем сравнительную таблицу ковариаций
    df_cov = pd.concat((X_df_cov, Z_df_cov), axis = 1)

    print ('Таблица №3 "Сравнение матрицы ковариации исходных и преобразованных признаков"')
    print(df_cov[columns_xz].fillna('-'))

    # сравним суммы дисперсий на исходном и преобразованном пространстве
    print(round(sum(X_train_dataframe.var()), 10), '- Сумма дисперсий исходных признаков')
    print(round(sum(Z_train_dataframe.var()), 10), '- Сумма дисперсий преобразованных признаков')
    print()
    print('Значимость компонент:')
    print(list(map(lambda x: round(x, 10), model_pca.explained_variance_ratio_)))

    # формируем таблицу корреляции исходных признаков
    X_df_cor = X_train_dataframe.corr().round(3)

    # формируем таблицу корреляции преобразованных признаков
    Z_df_cor = Z_train_dataframe.corr().round(3)

    # формируем сравнительную таблицу корреляции
    df_cor = pd.concat((X_df_cor, Z_df_cor), axis = 1)
    # df_cor.fillna(0)

    print ('Таблица №4 "Сравнение корреляции исходных и преобразованных признаков"')
    print(df_cor[columns_xz].fillna('-'))
    # print ('График №4 "Декорреляция нового признакового пространства"')
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    ax.grid(True, which='both')

    fig.suptitle('График №4 "Декорреляция нового признакового пространства"', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=1.05)

    ax.set_ylim(np.min(Z_train_norm[:, 1]) - 1, np.max(Z_train_norm[:, 1]) + 1)
    ax.set_xlim(np.min(Z_train_norm[:, 0]) - 1, np.max(Z_train_norm[:, 0]) + 1)
    ax.plot(Z_train_norm[:, 0], Z_train_norm[:, 1], 'o', color='green')
    plt.xlabel('1-я компонента')
    plt.ylabel('2-я компонента')
    #plt.show()

    # сохраним график в файл
    # fig.savefig('graph_4.png')