import csv
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:\Users\Ashok Mehta\Downloads\shuffle.csv')
y = data[0]
X = data.drop(data[0], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("\nX_train:\n")
print(X_train.head())
print(X_train.shape)

print("\nX_test:\n")
print(X_test.head())
print(X_test.shape)
print(len(X_train))
print(len(X_test))