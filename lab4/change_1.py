import pandas as pd

# Загружаю датасеты из файла
df_train = pd.read_csv('datasets/titanic_train.csv')
df_test = pd.read_csv('datasets/titanic_test.csv')

# Оставляю только нужные столбцы
df_train = df_train[['Pclass', 'Sex', 'Age']]
df_test = df_test[['Pclass', 'Sex', 'Age']]

# Записываю измененный датасет в файл
df_train.to_csv('datasets/titanic_train.csv', index=False)
df_test.to_csv('datasets/titanic_test.csv', index=False)
