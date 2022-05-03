

from numpy import NaN
import pandas as pd
#import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler

from ann_visualizer.visualize import ann_viz
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
import statsmodels.tsa.api as smt
import statsmodels.api as sm

import garchfunctions as garchfunctions
import filtering as filtering
import blackscholes as blackscholes

import model1 as model1
import callpricesummary as callpricesummary

import statsmodels.graphics.tsaplots as sgt




daatta1 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data1.csv", sep=";")
daatta2 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data2.csv", sep=";")
daatta3 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data3.csv", sep=";")
daatta4 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data4.csv", sep=";")
daatta5 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data5.csv", sep=";")
daatta6 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data6.csv", sep=";")
daatta7 = pd.read_csv(r"C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\data7.csv", sep=";")

#data for calculating volatility
df = pd.read_csv(r'C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\FTSE100.csv', sep=";")
rfdata = pd.read_csv(r'C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\rfdata.csv', sep=",")


rfdata["Name"] = pd.to_datetime(rfdata['Name'], format = '%Y-%m-%d')
rfdata["Name"] = rfdata["Name"].dt.strftime("%d.%m.%y")
rfdata["Name"] = pd.to_datetime(rfdata['Name'], format = "%d.%m.%y")
rfdata["RF"] = pd.to_numeric(rfdata["RF"],errors = 'coerce')
rfdata["RF"] = rfdata["RF"].div(100).round(3)




#print(rfdata)
#Calculating volatilities
df["FTSE 100 - PRICE INDEX"]=df["FTSE 100 - PRICE INDEX"].str.replace(',','.')
df["FTSE 100 - TOT RETURN IND"] = df["FTSE 100 - TOT RETURN IND"].str.replace(',','.')
df = df.rename(columns={"FTSE 100 - PRICE INDEX":"indeksi"})
df = df.rename(columns={"FTSE 100 - TOT RETURN IND":"totindeksi"})

df["indeksi"] = df.indeksi.astype("float")
df["totindeksi"] = df.totindeksi.astype("float")
window_size1m = 21
window_size3m = 63
window_size60d = 42
df["1mvol"] = df["indeksi"].pct_change().rolling(window_size1m).std()*(252**0.5)
df["3mvol"] = df["indeksi"].pct_change().rolling(window_size3m).std()*(252**0.5)
df["60dvol"] = df["indeksi"].pct_change().rolling(window_size60d).std()*(252**0.5)
df["MA"] = df["indeksi"].rolling(window = 100).mean()
df['vola'] = 100 * (df["indeksi"].pct_change())
df = df.dropna()
df['pct_change'] =  df['indeksi'].pct_change(1) * 100








#garchfunctions.plot_correlogram(df['vola'], lags=30, title='FTSE 100')
#plot_acf(df["vola"], lags = 30)
#plot_pacf(df["vola"], lags = 30)
#garch = garchfunctions.garch(df)

#undo the code to see garch plots
#df['vol_garch'].plot()
#plt.show()
print(df)

df = df.dropna()

#garch1 = arch_model(df['pct_change'], vol='GARCH', p = 1 , o = 0 , q = 1)
garch1 = arch_model(df['pct_change'], vol='GARCH', p=1, q=1, dist='normal')
#garch = arch_model(df['pct_change'], vol='GARCH', p=1, q=1, mean='constant')
fgarch = garch1.fit(disp='off') 

resid = fgarch.resid
st_resid = np.divide(resid, fgarch.conditional_volatility)
#ts_plot(resid, st_resid)
fgarch.summary()

df["garch"] = fgarch.resid



""""
def garchplots(df):
    df = df.dropna()
    plt.subplot(1, 2, 1)
    sm.graphics.tsa.plot_acf(df, lags=50)
    plt.xlabel('lags')
    plt.ylabel('corr')
    plt.show()
    plt.subplot(1, 2, 2)
    sm.graphics.tsa.plot_pacf(df, lags=50)
    plt.xlabel('lags')
    plt.ylabel('corr') 

    plt.show()


"""

#garchplots = garchplots(df["vol_garch"])
#garchfunctions.garchplots(df["vol_garch"])





daatta1 = pd.melt(daatta1,id_vars="Name")
daatta2 = pd.melt(daatta2,id_vars="Name")
daatta3 = pd.melt(daatta3,id_vars="Name")
daatta4 = pd.melt(daatta4,id_vars="Name")
daatta5 = pd.melt(daatta5,id_vars="Name")
daatta6 = pd.melt(daatta6,id_vars="Name")
daatta7 = pd.melt(daatta7,id_vars="Name")

#concatenating datas
combined_data = pd.concat([daatta1,daatta2,daatta3,daatta4,daatta5,daatta6,daatta7])
combined_data = combined_data[~combined_data.variable.str.contains('#ERROR')]



#splitting variable to A and B columns
combined_data[['A', 'B']] = combined_data['variable'].str.split(' - ', 1, expand=True)

#removing expirity year 2022 options
combined_data = combined_data[~combined_data.A.str.contains('22')]
#removing IFTS options
combined_data = combined_data[~combined_data.A.str.contains('IFTS')]
meltedframe = combined_data



meltedframe.loc[pd.isnull(meltedframe['B']) == True, 'B'] = "CALL_PRICE"
meltedframe.drop('variable', inplace=True, axis=1)
meltedframe.loc[meltedframe['B'].str.contains("OPT STRIKE PRICE"), 'B'] = "STRIKE"
meltedframe.loc[meltedframe['B'].str.contains("OPT.U/LYING PRICE"), 'B'] = "UNDERLYING"
meltedframe.loc[meltedframe['B'].str.contains("IMPLIED VOL."), 'B'] = "IV"


#Dropping duplicates and NA's
meltedframe = meltedframe.dropna()
meltedframe = meltedframe.drop_duplicates()
#long to wide
df1 = meltedframe.pivot(index=["Name","A"], columns="B",values="value").reset_index()
df1 = df1.dropna()
df1 = df1.drop_duplicates()
print(df1)
print("Number of unique options",df1["A"].nunique())




df1["UNDERLYING"]=df1["UNDERLYING"].str.replace(',','.')
df1["CALL_PRICE"] = df1["CALL_PRICE"].str.replace(',','.')
df1["STRIKE"] = df1["STRIKE"].str.replace(',','.')
df1["IV"]=df1["IV"].str.replace(',','.')



df1["UNDERLYING"] = df1.UNDERLYING.astype("float")
df1["STRIKE"] = df1.STRIKE.astype("float")
df1["CALL_PRICE"] = df1.CALL_PRICE.astype("float")
df1["IV"]=df1.IV.astype("float")
df1['Name']= pd.to_datetime(df1['Name'], format = '%d.%m.%Y')
df['Name']= pd.to_datetime(df['Name'], format = '%d.%m.%Y')


#moneyness
df1["Moneyness"] = df1["UNDERLYING"]/df1["STRIKE"] 
df1 = df1.sort_values(["A","Name"])



df1["paivat"] = df1.groupby("A").Name.transform('count')
df1["helper"] = df1.groupby("A").paivat.transform('cumcount')
df1["TTM"] = df1["paivat"] - df1["helper"]
df1["TTM"] = df1.TTM.astype("float")


df1 = filtering.filteringfunction(df1)

print("average ttm",df1["TTM"].mean())
df1["TTM2"] = df1["TTM"]
df1["TTM"] = df1["TTM"] / 252





#adding volatilities to the dataframe
df1 = df1.rename(columns={'Name_x': 'Name'})
df1 = df1.merge(df, on = "Name")
df1 = df1.merge(rfdata, on = "Name")



#calculating theoretical optionprices
df1["bsprice"] = blackscholes.blackscholes(df1["UNDERLYING"],df1["STRIKE"],df1["TTM"],0.01,df1["3mvol"])
df1["bsdelta"] = blackscholes.bsdelta(df1["UNDERLYING"],df1["STRIKE"],df1["TTM"],0.05,df1["3mvol"])
df1["bsprice"]=df1["bsprice"].round(4)
df1 = df1.dropna()
df1 = df1.drop_duplicates()

#extracting options that wont meet lower boundary condition
df1["lb"] = df1["UNDERLYING"] - df1["STRIKE"] *np.exp(-df1["RF"] * df1["TTM"])
df1 = df1[df1["CALL_PRICE"] > df1["lb"]]

print(df1)
prettyprint = callpricesummary.pricesummary(df1)
print(df1["TTM"].describe())
print(df1["TTM2"].describe())
print(df1.describe())

#normalizing data

"""
df1["UNDERLYING"] = df1["UNDERLYING"]/df1["STRIKE"]
df1["CALL_PRICE"] = df1["CALL_PRICE"]/df1["STRIKE"]
df1["bsprice"] = df1["bsprice"]/df1["STRIKE"]


"""

df1["CALL_PRICE2"] = df1["CALL_PRICE"]/df1["STRIKE"]
df1["bsprice2"] = df1["bsprice"]/df1["STRIKE"]




#nitializing
scaler=MinMaxScaler(feature_range=(0,1))
df1['STRIKE2'] = scaler.fit_transform(df1[["STRIKE"]])
df1['UNDERLYING2'] = scaler.fit_transform(df1[["UNDERLYING"]])
#print(df1)

#X=df1[['Moneyness', 'TTM', '60dvol', 'vol_garch']]
#y=df1['CALL_PRICE']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#y muuttuja = moneyness ja TTM
#x muuttuja = call price / stike price
df1 = df1.sort_values(by = ["A", "Name"])


n = 271689
n_train = (int)(0.8 * n)
train = df1[0:n_train]

"""
X_train = train[['UNDERLYING','Moneyness', 'TTM', '60dvol', 'vol_garch', 'RF']].values
y_train = train['CALL_PRICE'].values

"""

#X_train = train[['Moneyness', 'TTM', '60dvol', 'vol_garch', 'RF']].values
#X_train = train[['Moneyness', 'TTM']].values
X_train = train[['Moneyness', 'TTM','1mvol','3mvol','60dvol','RF', 'garch']].values
y_train = train['CALL_PRICE2'].values
test = df1[n_train+1:n]
print(len(test))


"""
X_test = test[['UNDERLYING','Moneyness', 'TTM', '60dvol', 'RF']].values
y_test = test['CALL_PRICE'].values
"""
#X_test = test[['Moneyness', 'TTM','60dvol', 'vol_garch', 'RF']].values
#X_test = test[['Moneyness', 'TTM']].values
X_test = test[['Moneyness', 'TTM','1mvol','3mvol','60dvol','RF','garch']].values
y_test = test['CALL_PRICE2'].values
#predicting with model1
#y_train_hat = model1.model1(X_train, y_train)

#checking accuracy
#x1acc = model1.CheckAccuracy(y_train, y_train_hat)
#x2acc = model1.CheckAccuracy(df1["CALL_PRICE"],df1["bsprice"])
def custom_activation(x):
    return backend.exp(x)



nodes = 120
#sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
model = Sequential()

model.add(Dense(nodes, input_dim=X_train.shape[1]))
model.add(LeakyReLU())
model.add(Dropout(0.25))

model.add(Dense(nodes, activation='elu'))
model.add(Dropout(0.25))

#model.add(Dense(nodes, activation='relu'))
model.add(Dense(nodes, activation='relu'))
model.add(Dropout(0.25))

#model.add(Dense(nodes, activation='elu'))
#model.add(Dense(nodes, activation='relu'))
#model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation(custom_activation))
#model.add(Activation('softplus'))

model.compile(loss='mse', optimizer='rmsprop')

# fitting neural network
model.fit(X_train, y_train, batch_size=128,

epochs=10, validation_split=0.1, verbose=2)


plot_model(model, to_file = "neuralplot.png")

#y_train_hat = model.predict(X_train)
# reduce dim (240000,1) -> (240000,) to match y_train's dim
#y_train_hat = np.squeeze(y_train_hat)


def CheckAccuracy(y,y_hat):
    print("moi")
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

    plt.title("Test sample diagnostics")
    plt.subplot(1, 2, 1)
    #plt.figure(figsize=(14,10))
    plt.title("Predicted vs Actual")
    plt.scatter(y, y_hat,color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Actual Price C/K Test Sample')
    plt.ylabel('Predicted Price C/K Test Sample') 
    #plt.xlim([0.0, 0.25])
    #plt.ylim([0.0, 0.25])
    #plt.show()
    
    plt.subplot(1, 2, 2)
    #plt.figure(figsize=(14,10))
    plt.title("Absolute error by option price size")
    plt.ylabel('Absolute Error')
    plt.xlabel('C/K')
    plt.scatter(y ,abs(stats['diff']), alpha=0.4, s=0.5,color='black')
    plt.show()




"""
def testiplot(datasetti2):
    plt.figure(figsize=(14,10))
    plt.scatter(datasetti2["CALL_PRICE"],datasetti2["bsprice"],color='black')
    plt.show()

"""
#CheckAccuracy(df1["CALL_PRICE"], df1["bsprice"])


#in-sample
y_train_hat = model.predict(X_train)
y_train_hat = np.squeeze(y_train_hat)
CheckAccuracy(y_train, y_train_hat)

#test["ANNpred"] = y_train_hat.tolist()


#print(test)

#out-sample
y_test_hat = model.predict(X_test)
y_test_hat = np.squeeze(y_test_hat)



test_stats = CheckAccuracy(y_test, y_test_hat)


test["difference"] = y_test - y_test_hat

plt.hist(test["difference"], bins = 50, color = "black")
plt.title("Absolute error histogram")
plt.ylabel('Density')
plt.xlabel('Absolute Error')
plt.show()





test["ANNpred"] = y_test_hat.tolist()
print(test)


gradients = tf.gradients(model.output[:, 0], model.input)
evaluated_gradients_1 = sess.run(gradients[0], feed_dict={model.input: 
X_test})
print(evaluated_gradients_1)


def Extract(lst):
    return [item[0] for item in lst]


deltaANN = Extract(evaluated_gradients_1)
test["deltaANN"] = deltaANN


test["annvalue"] = test["ANNpred"] * test["STRIKE"] 
print(test["deltaANN"].max())

print(test)

xx = test.to_csv("moneynesstest.csv")


#CheckAccuracy(test["CALL_PRICE2"], test["bsprice2"])


"""
suuri = test[test["TTM2"] > 180]
pieni = test[test["TTM2"] < 60]
normi = test[test["TTM2"] > 60]
normi = test[test["TTM2"] < 180]

print("tarkastelu")
CheckAccuracy(suuri["CALL_PRICE2"],suuri["ANNpred"])
print("^suuri")

CheckAccuracy(pieni["CALL_PRICE2"],pieni["ANNpred"])
print("^pieni")


CheckAccuracy(normi["CALL_PRICE2"],normi["ANNpred"])
print("^normi")
"""
print("garch")
#CheckAccuracy(train["CALL_PRICE2"], train["bsprice2"])
def plotting_function_ANN(df):
    
    plt.subplot(2, 2, 1)
    #plt.subplot(1, 2, 1)
    plt.title("Moneyness vs Pricing Error")
    df['pricing_error_ANN'] = df["ANNpred"] - df["CALL_PRICE2"]
    plt.scatter(df["pricing_error_ANN"], df["Moneyness"], color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Pricing Error (ANN_garch)')
    plt.ylabel('Moneyness')
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    #plt.subplot(1, 2, 2)
    plt.title("Predicted vs Actual")
    plt.scatter(df["CALL_PRICE2"], df["ANNpred"],color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Actual C/K ')
    plt.ylabel('Predicted C/K')
    plt.tight_layout()
    #plt.show()
    plt.subplot(2, 2, 2)
    #plt.subplot(1, 1, 2)
    plt.hist(df['pricing_error_ANN'], bins = 50, color = "black")
    plt.title("Absolute error histogram")
    plt.ylabel('Density')
    plt.xlabel('Absolute Error')
    plt.tight_layout()
    plt.show()

plotting_function_ANN(test)