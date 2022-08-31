#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pandas as pd
#import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si
import matplotlib.dates as mdates
df = pd.read_csv(r'C:\Users\q8606\Desktop\GRADUTUTKIMUKSET\FTSE100.csv',
                 sep=";", index_col='Name', parse_dates=True, dayfirst=True)
#df["Name"] = pd.to_datetime(df['Name'], format = '%d.%m.%Y')
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
window_size5d = 5
df["1mvol"] = df["indeksi"].pct_change().rolling(
    window_size1m).std()*(252**0.5)
df["3mvol"] = df["indeksi"].pct_change().rolling(
    window_size3m).std()*(252**0.5)
df["60dvol"] = df["indeksi"].pct_change().rolling(
    window_size60d).std()*(252**0.5)
df["5dvol"] = df["indeksi"].pct_change().rolling(
    window_size5d).std()*(252**0.5)
df["MA"] = df["indeksi"].rolling(window=100).mean()
df['vola'] = 100 * (df["indeksi"].pct_change())
df = df.dropna()


def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


def garch(df):
    """Calculates GARCH based on the underlying asset pandas dataframe"""
    r = df['vola']
    garch11 = arch_model(r, p=1, q=1, rescale=True)
    res = garch11.fit(update_freq=10)
    print(res.summary())

    #sigma =  0.01 *np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]']) * np.sqrt(252)
    sigma = 0.01 * np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid **
                           2 + shift(res.resid, 1)**2 * res.params['beta[1]']) * np.sqrt(252)

    df['vol_garch'] = sigma
    return(df)


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values),2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)


garch(df)
df = df.dropna()

print(df)


def plotting_index(df):

    fig, ax = plt.subplots(figsize=(10, 5))
    #ax.figure(figsize=(10, 10))
    ax.plot(df.index, df['60dvol'], color="black", label="60-day-volatility")
    ax.set_xlabel("year")
    ax.set_ylabel("volatility level")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate(rotation=30)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.plot(df.index, df['1mvol'], color="cyan", label="21-day-volatility")
    ax.plot(df.index, df['3mvol'], color="magenta", label="90-day-volatility")
    ax.legend()
    plt.show()


def plotting_garchvol(df):

    fig, ax = plt.subplots(figsize=(10, 5))
    #ax.figure(figsize=(10, 10))
    ax.set_xlabel("year")
    ax.set_ylabel("volatility level")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate(rotation=30)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.plot(df.index, df['vol_garch'], color="black", label="GARCH(1,1)")
    # ax.legend(["GARCH(1,1)"])
    ax.plot(df.index, df['5dvol'], color="red", label="5-day-volatility")
    ax.legend()
    plt.show()


print(df)
plotting_index(df)
plotting_garchvol(df)
print(df.describe())


# In[ ]:
