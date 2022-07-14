
import os
import pandas as pd
import glob
import datetime as datetime

from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import numpy as np


def read_files(PATH):
    files = os.listdir(PATH)
    lista = []
    for file in files:

        df = pd.read_csv(os.path.join(PATH, file),
                         sep=";")
        df = pd.melt(df, id_vars="Name")
        lista.append(df)
        
    final_df = pd.concat(lista, axis=0, ignore_index=True)
    return final_df


def manipulate_rf_data(rfdata):

    rfdata = rfdata.assign(Name=pd.to_datetime(rfdata['Name'], format='%Y-%m-%d'), errors='coerce',
                           RF=pd.to_numeric(rfdata["RF"], errors='coerce'))
    return rfdata


def manipulate_option_data(optiondata):
    
    #df = optiondata.melt(id_vars="Name")
    df = optiondata
    df = df[~df.variable.str.contains('#ERROR')]
    df[['A', 'B']] = df['variable'].str.split(' - ', 1, expand=True)
    df = df[~df.A.str.contains('22')]
    df = df[~df.A.str.contains('IFTS')]


    df.loc[pd.isnull(df['B']) == True, 'B'] = "CALL_PRICE"
    df.drop('variable', inplace=True, axis=1)
    df.loc[df['B'].str.contains("OPT STRIKE PRICE"), 'B'] = "STRIKE"
    df.loc[df['B'].str.contains("OPT.U/LYING PRICE"), 'B'] = "UNDERLYING"
    df.loc[df['B'].str.contains("IMPLIED VOL."), 'B'] = "IV"
    
    df = df.dropna()
    df = df.drop_duplicates()
    pivot_df = df.pivot(index=["Name", "A"],
                        columns="B", values="value").reset_index()
   #df1 = meltedframe.pivot(index=["Name","A"], columns="B",values="value").reset_index()
    df = pivot_df.assign(UNDERLYING=pivot_df["UNDERLYING"].str.replace(',', '.'),
                   CALL_PRICE=pivot_df["CALL_PRICE"].str.replace(',', '.'),
                   STRIKE=pivot_df["STRIKE"].str.replace(',', '.'),
                   IV=pivot_df["IV"].str.replace(',', '.'),
                   )
    df = df.assign(UNDERLYING=df.UNDERLYING.apply(lambda x: float(x)),
                   STRIKE=df.STRIKE.apply(lambda x: float(x)),
                   CALL_PRICE=df.CALL_PRICE.apply(lambda x: float(x)),
                   IV=df.IV.apply(lambda x: float(x)))
    df["Moneyness"] = df["UNDERLYING"]/df["STRIKE"]
    df = df.sort_values(["A", "Name"])

    df["paivat"] = df.groupby("A").Name.transform('count')
    df["helper"] = df.groupby("A").paivat.transform('cumcount')
    df["TTM"] = df["paivat"] - df["helper"]
    df["TTM"] = df.TTM.astype("float")
    df["TTM2"] = df["TTM"]
    df["TTM"] = df["TTM"] / 252
    
    return df


def vol_data():
    df = pd.read_csv(
        r'C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\FTSE100.csv', sep=";")
    df["FTSE 100 - PRICE INDEX"] = df["FTSE 100 - PRICE INDEX"].str.replace(
        ',', '.')
    df["FTSE 100 - TOT RETURN IND"] = df["FTSE 100 - TOT RETURN IND"].str.replace(
        ',', '.')
    df = df.rename(columns={"FTSE 100 - PRICE INDEX": "indeksi"})
    df = df.rename(columns={"FTSE 100 - TOT RETURN IND": "totindeksi"})

    df["indeksi"] = df.indeksi.astype("float")
    df["totindeksi"] = df.totindeksi.astype("float")
    window_size1m = 21
    window_size3m = 63
    window_size60d = 42
    df["1mvol"] = df["indeksi"].pct_change().rolling(
        window_size1m).std()*(252**0.5)
    df["3mvol"] = df["indeksi"].pct_change().rolling(
        window_size3m).std()*(252**0.5)
    df["60dvol"] = df["indeksi"].pct_change().rolling(
        window_size60d).std()*(252**0.5)
    df["MA"] = df["indeksi"].rolling(window=100).mean()
    df['vola'] = 100 * (df["indeksi"].pct_change())
    df = df.dropna()
    df['pct_change'] = df['indeksi'].pct_change(1) * 100
    df = df.dropna()
    garch1 = arch_model(df['pct_change'], vol='GARCH', p=1, q=1, dist='normal')
    fgarch = garch1.fit(disp='off')
    resid = fgarch.resid
    st_resid = np.divide(resid, fgarch.conditional_volatility)
    fgarch.summary()

    df["garch"] = fgarch.resid
    return df
