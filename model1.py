from sklearn.preprocessing import MinMaxScaler
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
import matplotlib.pyplot as plt

def custom_activation(x):
    return backend.exp(x)


def model1(X_train, y_train):



    nodes = 120
    model = Sequential()

    model.add(Dense(nodes, input_dim=X_train.shape[1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))

    model.add(Dense(nodes, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nodes, activation='elu'))
    model.add(Dropout(0.25))

    model.add(Dense(1))
    model.add(Activation(custom_activation))

    model.compile(loss='mse', optimizer='rmsprop')

# fitting neural network
    model.fit(X_train, y_train, batch_size=64,
    epochs=10, validation_split=0.1, verbose=2)
#model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.1, verbose=2)
    y_train_hat = model.predict(X_train)
# reduce dim (240000,1) -> (240000,) to match y_train's dim
    y_train_hat = np.squeeze(y_train_hat)

    
    return y_train_hat


def CheckAccuracy(y, y_hat):
    stats = dict()

    stats['diff'] = y - y_hat

    stats['mse'] = np.mean(stats['diff']**2)
    print("Mean Squared Error:      ", stats['mse'])

    stats['rmse'] = np.sqrt(stats['mse'])
    print("Root Mean Squared Error: ", stats['rmse'])

    stats['mae'] = np.mean(abs(stats['diff']))
    print("Mean Absolute Error:     ", stats['mae'])

    stats['mpe'] = np.sqrt(stats['mse'])/np.mean(y)
    print("Mean Percent Error:      ", stats['mpe'])

    plt.scatter(y, y_hat, color='black', linewidth=0.3, alpha=0.4, s=0.5)
    plt.xlabel('Actual Price', fontsize=20, fontname='Times New Roman')
    plt.ylabel('Predicted Price', fontsize=20, fontname='Times New Roman')
    plt.show()

    plt.hist(stats['diff'], bins=50, edgecolor='black', color='white')
    plt.xlabel('Diff')
    plt.ylabel('Density')
    plt.show()

#testataan 123"


