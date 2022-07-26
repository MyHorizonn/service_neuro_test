import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import typing
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Инициализация слоев сетки
        self.conv1 = nn.Conv1d(20, 1024, 2, 2, 1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv1d(1024, 512, 2, 2, 1)
        self.conv2_drop = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fcl1 = nn.Linear(41472, 1024)
        self.fcl3 = nn.Linear(1024, 512)
        self.fcl4 = nn.Linear(512, 100)
        self.fcl5 = nn.Linear(100, 6)

    def forward(self, x):
        # Реализация функции прохода по сетке
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fcl1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fcl3(x))
        x = F.relu(self.fcl4(x))
        x = F.relu(self.fcl5(x))
        return x

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, eval_set: typing.Tuple[torch.Tensor, torch.Tensor], iters: int, loss_func, optimizer, device: torch.device):
        """
        X_train, y_train - training set
        eval_set - validation set
        iters - number of iterations
        loss_func - loss function
        optimizer - optimization alghoritm
        device - index of GPU device
        """
        # Проверка размерности тензоров
        assert X_train.shape[0] == y_train.shape[0], (
            'Incorrect X_train and y_train shapes')
        assert eval_set[0].shape[0] == eval_set[1].shape[0], (
            'Incorrect X_test and y_test shapes')
        assert X_train.shape[1] == eval_set[0].shape[1], (
            'Incorrect X_train and X_test shapes')

        # eval_set кортеж валидационных данных (X, y)
        train_x, train_y = X_train, y_train
        test_x, test_y = eval_set[0], eval_set[1]
        # Привожу типы тензоров в LongTensor
        train_y = train_y.type(torch.LongTensor)
        test_y = test_y.type(torch.LongTensor)

        # Размещение тензоров на GPU для тренировки
        train_x, train_y = train_x.to(device), train_y.to(device)
        test_x, test_y = test_x.to(device), test_y.to(device)

        # Инициализация переменных для фиксирования лучших итераций на тренировачных/тестовых данных
        best_train = 100
        best_train_iter = 0
        best_test = 100
        best_test_iter = 0

        # Начало процесса обучения сетки
        for iter in range(1, iters+1):

            # Устанавливается режим тренировки модели
            self.train()

            # Обнуление градиентов
            optimizer.zero_grad()

            # Проход данных по сетке
            train_output = self(train_x)

            # Вычисление функции ошибки
            train_loss = loss_func(train_output, train_y)

            # Вычисление градиентов параметров
            train_loss.backward()
            # Оптимизация параметров
            optimizer.step()
            train_loss = train_loss.item()

            # Проверка на лучшую итерацию на тренировачных данных
            if train_loss < best_train:
                best_train = train_loss
                torch.save(self.state_dict(), 'best-model-parameters_train.pt')
                best_train_iter = iter

            # Устанавливается режим тестирования модели
            self.eval()
            # torch.no_grad() для того чтобы не вычислять градиенты и не проиводить лишних операций
            with torch.no_grad():
                test_output = self(test_x)
                test_loss = loss_func(test_output, test_y)
                test_loss = test_loss.item()

            # Проверка на лучшую итерацию на тестовых данных
            if test_loss < best_test:
                best_test = test_loss
                torch.save(self.state_dict(), 'best-model-parameters_test.pt')
                best_test_iter = iter

            # Логирование обучения и тестирования
            if iter % 10 == 0:
                print(f'Epoch: {iter}', '\t', '\t',
                      f'train_loss: {train_loss:.5f}', '\t', '\t', f'test_loss: {test_loss:.5f}',
                      '\t', f'Best train: {best_train:.5f}', '\t', f'Best_train_epoch: {best_train_iter}',
                      '\t', f'Best test: {best_test:.5f}', '\t', f'Best_test_epoch: {best_test_iter}')
