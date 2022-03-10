import pandas as pd
import numpy as np




def filteringfunction(df):
    
    # deep in- or deep out-of-the-money extraction
    df = df[df["Moneyness"] >= 0.85]
    df = df[df["Moneyness"] <= 1.15]
    df = df[df["CALL_PRICE"] >= 10]

    #Option has less than 15 days to maturity
    df = df[df["TTM"] >= 15]
    df = df[df["TTM"] <= 252]
    return df
