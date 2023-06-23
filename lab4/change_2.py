import pandas as pd

# Загружаю датасетвы из файла
df_train = pd.read_csv('datasets/titanic_train.csv')
df_test = pd.read_csv('datasets/titanic_test.csv')

# Заполняю пропущенные значения возраста средним значением
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

# Записываю измененный датасет в файл
df_train.to_csv('datasets/titanic_train.csv', index=False)
df_test.to_csv('datasets/titanic_test.csv', index=False)
