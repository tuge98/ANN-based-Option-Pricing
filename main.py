
from py_files.hedging_scenario import hedging_func
from py_files.neural_nets import MLPnetwork
from py_files.filtering import filteringfunction
import py_files.blackscholes as blackscholes
import py_files.garchfunctions as garchfunctions
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from scipy.stats import probplot, moment
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import os
import pandas as pd
import glob
import datetime as datetime


from py_files.datahandling import read_files, manipulate_rf_data, manipulate_option_data, vol_data
from numpy import NaN
import pandas as pd
#import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    PATH = r'C:\Users\q8606\Desktop\DATA__GRADU'
    rfdata = pd.read_csv(
        r'C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\rfdata.csv', sep=",")
    df = read_files(PATH)
    df = manipulate_option_data(df)
    index_data = vol_data()

    df = df.rename(columns={'Name_x': 'Name'})
    df = df.merge(index_data, on="Name")

    #df = datawithvols.merge(rfdata, on = "Name")
    df["bsprice"] = blackscholes.blackscholes(
        df["UNDERLYING"], df["STRIKE"], df["TTM"], 0.01, df["3mvol"])
    df["bsdelta"] = blackscholes.bsdelta(
        df["UNDERLYING"], df["STRIKE"], df["TTM"], 0.01, df["3mvol"])
    df["bsprice"] = df["bsprice"].round(4)
    df = df.dropna()
    df = df.drop_duplicates()

    df1 = filteringfunction(df)
    df1["lb"] = df1["UNDERLYING"] - df1["STRIKE"] * np.exp(-0.01 * df1["TTM"])
    df1 = df1[df1["CALL_PRICE"] > df1["lb"]]

    df1["CALL_PRICE2"] = df1["CALL_PRICE"]/df1["STRIKE"]
    df1["bsprice2"] = df1["bsprice"]/df1["STRIKE"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1['STRIKE2'] = scaler.fit_transform(df1[["STRIKE"]])
    df1['UNDERLYING2'] = scaler.fit_transform(df1[["UNDERLYING"]])
    df1 = df1.sort_values(by=["A", "Name"])

    n = 95589
    #n = 6647
    n_train = (int)(0.8 * n)
    train = df1[0:n_train]
    X_train = train[['Moneyness', 'TTM', '1mvol',
                     '3mvol', '60dvol', 'garch']].values
    y_train = train['CALL_PRICE2'].values
    test = df1[n_train+1:n]
    X_test = test[['Moneyness', 'TTM', '1mvol',
                   '3mvol', '60dvol', 'garch']].values
    y_test = test['CALL_PRICE2'].values
    model = MLPnetwork(num_epochs=1000, batch=64, nodes=120,
                       X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    fit = model.initNetwork()
    y_test_hat = model.outSample()
    NNdelta = model.obtainGradients()
    test["deltaANN"] = NNdelta
    hedging_func(test)
    import IPython
    IPython.embed()
