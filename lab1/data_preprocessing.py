import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загружаю тренировочную и тестовую выборки c шумом и без
train_data = pd.read_csv('train/data.csv')
test_data = pd.read_csv('test/data.csv')

train_data_noise = pd.read_csv('train/data_noise.csv')
test_data_noise = pd.read_csv('test/data_noise.csv')

# Выполняю предобработку данных
scaler = StandardScaler().fit(train_data[['x', 'y']])
train_data[['x', 'y']] = scaler.transform(train_data[['x', 'y']])
scaler = StandardScaler().fit(test_data[['x', 'y']])
test_data[['x', 'y']] = scaler.transform(test_data[['x', 'y']])

scaler = StandardScaler().fit(train_data_noise[['x_noise', 'y_noise']])
train_data_noise[['x_noise', 'y_noise']] = scaler.transform(train_data_noise[['x_noise', 'y_noise']])
scaler = StandardScaler().fit(test_data_noise[['x_noise', 'y_noise']])
test_data_noise[['x_noise', 'y_noise']] = scaler.transform(test_data_noise[['x_noise', 'y_noise']])

# Записываю предобработанные данные в соответствующие файлы
train_data.to_csv('train/data_preprocessing.csv', index=False)
test_data.to_csv('test/data_preprocessing.csv', index=False)
train_data_noise.to_csv('train/data_noise_preprocessing.csv', index=False)
test_data_noise.to_csv('test/data_noise_preprocessing.csv', index=False)