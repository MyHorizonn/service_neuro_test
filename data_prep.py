import librosa
import numpy as np
from numpy import lib
import torch


def data_preprocessing(fname, test_path, fon_train, micout_train, n_mfcc: int = 20):

    # Загрузка данных
    data, rate = librosa.load(fname)
    fon, _ = librosa.load(fon_train)
    mic_out, _ = librosa.load(micout_train)

    # Разделение данных на классы для дальнейшего изменения размерности и создания лейблов классов
    psk50 = data[50*rate:290*rate].copy()
    psk100 = data[350*rate:680*rate].copy()
    post_noise = data[750*rate:900*rate].copy()
    voice = data[925*rate:1015*rate].copy()

    # Изменение размерности так чтобы каждый сэмпл класса длился 30 секунд или 661500 кадров (как в тестовых данных)
    fon = fon.reshape((fon.shape[0] // rate // 30, rate * 30))
    mic_out = mic_out.reshape((mic_out.shape[0] // rate // 30, rate * 30))
    psk50 = psk50.reshape((psk50.shape[0] // rate // 30, rate * 30))
    psk100 = psk100.reshape((psk100.shape[0] // rate // 30, rate * 30))
    post_noise = post_noise.reshape(
        (post_noise.shape[0] // rate // 30, rate * 30))
    voice = voice.reshape((voice.shape[0] // rate // 30, rate * 30))

    # Объединение всех сэмплов в тренировочные данные и создание лейблов классов
    # Для избежания дисбаланса классов, брал часть сэмплов доминирующих по количеству классов
    X_train = np.concatenate(
        (fon[:5], mic_out, psk50[:5], psk100[:5], post_noise, voice))
    y_train = np.array([2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 0,
                       0, 0, 0, 0, 5, 5, 5, 5, 5, 1, 1, 1])

    # Загрузка тестовых данных
    test_fon, _ = librosa.load(test_path + 'test_fon.wav')
    test_golosa, _ = librosa.load(test_path + 'test_golosa.wav')
    test_psk50, _ = librosa.load(test_path + 'test_psk50.wav')
    test_psk100, _ = librosa.load(test_path + 'test_psk100.wav')
    test_mic_out, _ = librosa.load(test_path + 'test_microphoneout.wav')
    test_post_noise, _ = librosa.load(test_path + 'test_postshum.wav')

    # Изменение размерности (все тестовые данные длятся 30 секунд, однако у некоторых классов разное количество кадров, поэтому приходилось обрезать (до 661500 кадров))
    test_fon = test_fon[:X_train.shape[1]]
    test_golosa = test_golosa[:X_train.shape[1]]
    test_psk50 = test_psk50[:X_train.shape[1]]
    test_psk100 = test_psk100[:X_train.shape[1]]
    test_mic_out = test_mic_out[:X_train.shape[1]]
    test_post_noise = test_post_noise[:X_train.shape[1]]

    # Объединение всех сэмплов в тестовые данные и создание лейблов классов
    X_test = np.concatenate(
        (test_fon, test_golosa, test_psk50, test_psk100, test_mic_out, test_post_noise))
    y_test = np.array([2, 1, 4, 0, 3, 5])

    # reshape test data
    X_test = X_test.reshape(6, 661500)

    # my_mfcc обертка функции librosa.features.mfcc для вычисления мел-кепстральных коэффициентов матриц
    X_train, X_test = my_mfcc(X_train, X_test, rate, n_mfcc)

    # Измение типа на torch.Tensor для torch модели
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    return X_train, X_test, y_train, y_test


def my_mfcc(train, test, rate, n_mfcc: int = 20):

    mfcc_train = list()
    mfcc_test = list()

    for i in range(train.shape[0]):
        mfcc_train.append(librosa.feature.mfcc(
            np.array(train[i]), sr=rate, n_mfcc=n_mfcc))

    for k in range(test.shape[0]):
        mfcc_test.append(librosa.feature.mfcc(
            np.array(test[k]), sr=rate, n_mfcc=n_mfcc))

    return np.array(mfcc_train), np.array(mfcc_test)
