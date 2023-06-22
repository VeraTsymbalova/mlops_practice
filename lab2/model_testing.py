import os
import pickle
import pandas as pd

test_data = pd.read_csv('test/data_preprocessing.csv')
test_data_noise = pd.read_csv('test/data_noise_preprocessing.csv')

# Загружаю модель и оцениваю ее результативность на тестовой выборке
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
metric_value = model.score(test_data[['x']], test_data['y'])
print(f"Точность модели на данных без шума: {metric_value:.3f}")

with open('model_noise.pkl', 'rb') as f:
    model_noise = pickle.load(f)
metric_value = model_noise.score(test_data_noise[['x_noise']], test_data_noise['y_noise'])
print(f"Точность модели на данных с шумом: {metric_value:.3f}")