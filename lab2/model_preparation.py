import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Загружаю предобработанные данные
train_data = pd.read_csv('train/data_preprocessing.csv')
train_data_noise = pd.read_csv('train/data_noise_preprocessing.csv')

# Обучаю модель и сохраняю результаты с помощью модуля pickle
model = LinearRegression().fit(train_data[['x']], train_data['y'])
pickle.dump(model, open('model.pkl', 'wb'))

model_noise = LinearRegression().fit(train_data_noise[['x_noise']], train_data_noise['y_noise'])
pickle.dump(model_noise, open('model_noise.pkl', 'wb'))