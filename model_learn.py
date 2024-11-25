import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


iris = load_iris()
X = iris.data
y = iris.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


print(len(iris))
print(len(y_train_one_hot))
print(len(y_test_one_hot))

loss_function = 'mean_squared_error'


model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(loss=loss_function, optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train_one_hot, epochs=250, batch_size=10, verbose=1)