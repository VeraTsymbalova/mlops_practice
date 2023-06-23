import pandas as pd

# Загружаю датасетвы из файла
df_train = pd.read_csv('datasets/titanic_train.csv')
df_test = pd.read_csv('datasets/titanic_test.csv')

# Создаю новый признак с использованием one-hot-encoding для строкового признака “Пол”.
sex_dummies_train = pd.get_dummies(df_train.Sex)
sex_dummies_test = pd.get_dummies(df_test.Sex)
df_train = pd.concat([df_train, sex_dummies_train], axis=1)
df_test = pd.concat([df_test, sex_dummies_test], axis=1)

# Записываю измененный датасет в файл
df_train.to_csv('datasets/titanic_train.csv', index=False)
df_test.to_csv('datasets/titanic_test.csv', index=False)
