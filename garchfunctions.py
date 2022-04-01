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


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    sigma =  0.01 *np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + shift(res.resid, 1)**2 * res.params['beta[1]']) * np.sqrt(252)
    
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






def garchplots(df):
    sm.graphics.tsa.plot_acf(df, lags=200)
    plt.xlabel('lags')
    plt.ylabel('corr')
   
    sm.graphics.tsa.plot_pacf(df, lags=200)
    plt.xlabel('lags')
    plt.ylabel('corr') 

    plt.show()