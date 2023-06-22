import pandas as pd

df_train = pd.read_csv('datasets/titanic_train.csv')
df_test = pd.read_csv('datasets/titanic_test.csv')

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())

df_train.to_csv('datasets/titanic_train.csv', index=False)
df_test.to_csv('datasets/titanic_test.csv', index=False)
