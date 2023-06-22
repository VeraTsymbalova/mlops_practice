import os
import random
import numpy as np
import pandas as pd


data_file = 'data.csv' #для записи данных без шума
data_file_noise = 'data_noise.csv' #для записи данных с шумом
train_data_folder = 'train'
test_data_folder = 'test'

# Проверяю существуют ли папки для тернировочных и тестовых выборок, если нет, создадаю их
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('test'):
    os.mkdir('test')

# Генерирую данные без шума
x_min = 0
x_max = 1000
x = np.array([random.randint(x_min, x_max) for i in range(100)])

b_0 = 2
b_1 = 12
y = b_0 * x + b_1

# Генерирую данные c шумом
x_noise = np.array([random.randint(x_min, x_max) for i in range(100)])
noise = 150
e = np.array([random.randint(-noise, noise) for i in range(100)])

b_0 = 2
b_1 = 12
y_noise = b_0 * x_noise + b_1 + e

# Объединяю x и y в датафрейм
data = pd.DataFrame({'x': x, 'y': y})
data_noise = pd.DataFrame({'x_noise': x_noise, 'y_noise': y_noise})

# Разбиваю данные на тренировочную и тестовую выборки
train_size = int(0.8 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

train_size_noise = int(0.8 * len(data_noise))
train_data_noise = data_noise.iloc[:train_size]
test_data_noise = data_noise.iloc[train_size:]

# Записываю выборки в соответствующие файл
train_data.to_csv('train/data.csv', index=False)
test_data.to_csv('test/data.csv', index=False)
train_data_noise.to_csv('train/data_noise.csv', index=False)
test_data_noise.to_csv('test/data_noise.csv', index=False)