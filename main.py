

from numpy import NaN
import pandas as pd
#import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si

from sklearn.preprocessing import MinMaxScaler
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








garchfunctions.plot_correlogram(df['vola'], lags=100, title='FTSE 100')
garch = garchfunctions.garch(df)

#undo the code to see garch plots
df['vol_garch'].plot()
plt.show()
print(df)






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

prettyprint = callpricesummary.pricesummary(df1)
df1 = filtering.filteringfunction(df1)

print("average ttm",df1["TTM"].mean())
df1["TTM"] = df1["TTM"] / 252





#adding volatilities to the dataframe
df1 = df1.rename(columns={'Name_x': 'Name'})
df1 = df1.merge(df, on = "Name")
df1 = df1.merge(rfdata, on = "Name")



#calculating theoretical optionprices
df1["bsprice"] = blackscholes.blackscholes(df1["UNDERLYING"],df1["STRIKE"],df1["TTM"],df1["RF"],df1["60dvol"])
df1["bsprice"]=df1["bsprice"].round(4)
df1 = df1.dropna()
df1 = df1.drop_duplicates()

#extracting options that wont meet lower boundary condition
df1["lb"] = df1["UNDERLYING"] - df1["STRIKE"] *np.exp(-df1["RF"] * df1["TTM"])
df1 = df1[df1["CALL_PRICE"] > df1["lb"]]
print(df1)



#normalizing data
df1["UNDERLYING"] = df1["UNDERLYING"]/df1["STRIKE"]
df1["CALL_PRICE"] = df1["CALL_PRICE"]/df1["STRIKE"]
df1["bsprice"] = df1["bsprice"]/df1["STRIKE"]




#nitializing
scaler=MinMaxScaler(feature_range=(0,1))
#df1['MA'] = scaler.fit_transform(df1[["MA"]])
print(df1)

X=df1[['Moneyness', 'TTM', '60dvol', 'vol_garch']]
y=df1['CALL_PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



#predicting with model1
y_train_hat = model1.model1(X_train, y_train)

#checking accuracy
x1acc = model1.CheckAccuracy(y_train, y_train_hat)
#x2acc = model1.CheckAccuracy(df1["CALL_PRICE"],df1["bsprice"])



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
    
    plt.scatter(y, y_hat,color='black',linewidth=0.3,alpha=0.4, s=0.5)
    plt.xlabel('Actual Price',fontsize=20,fontname='Times New Roman')
    plt.ylabel('Predicted Price',fontsize=20,fontname='Times New Roman') 
    plt.show()
    
    
    plt.hist(stats['diff'], bins=50,edgecolor='black',color='white')
    plt.xlabel('Diff')
    plt.ylabel('Density')
    plt.show()
CheckAccuracy(df1["CALL_PRICE"],df1["bsprice"])