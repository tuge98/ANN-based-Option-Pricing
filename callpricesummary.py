import pandas as pd



from tabulate import tabulate
import numpy as np



def pricesummary(df):

        
    
    A = np.mean(df.CALL_PRICE[ (df['TTM2'] < 60) & (df['Moneyness'] < 0.95)])
    B = np.mean(df.CALL_PRICE[ (df['TTM2'] > 60) & (df['TTM2'] < 180) & (df['Moneyness'] < 0.95)])
    C = np.mean(df.CALL_PRICE[ (df['TTM2'] > 180) & (df['Moneyness'] < 0.95)])
    D = np.mean(df.CALL_PRICE[ (df['Moneyness'] < 0.95)])

    E = np.mean(df.CALL_PRICE[ (df['TTM2'] < 60) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])
    F = np.mean(df.CALL_PRICE[ (df['TTM2'] > 60) & (df['TTM2'] < 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])
    G = np.mean(df.CALL_PRICE[ (df['TTM2'] > 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])
    H = np.mean(df.CALL_PRICE[ (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])

    I = np.mean(df.CALL_PRICE[ (df['TTM2'] < 60) & (df['Moneyness'] >= 1.05)])
    J = np.mean(df.CALL_PRICE[ (df['TTM2'] > 60) & (df['TTM2'] < 180) & (df['Moneyness'] >= 1.05)])
    K = np.mean(df.CALL_PRICE[ (df['TTM2'] > 180) & (df['Moneyness'] >= 1.05)])
    L = np.mean(df.CALL_PRICE[ (df['Moneyness'] >= 1.05)])

    M = (df.CALL_PRICE[ (df['TTM2'] < 60) & (df['Moneyness'] < 0.95)]).count()
    N = (df.CALL_PRICE[ (df['TTM2'] > 60) & (df['TTM2'] < 180) & (df['Moneyness'] < 0.95)]).count()
    O = (df.CALL_PRICE[ (df['TTM2'] > 180) & (df['Moneyness'] < 0.95)]).count()
    P = (df.CALL_PRICE[ (df['Moneyness'] < 0.95)]).count()

    Q = (df.CALL_PRICE[ (df['TTM2'] < 60) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()
    R = (df.CALL_PRICE[ (df['TTM2'] > 60) & (df['TTM2'] < 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()
    S = (df.CALL_PRICE[ (df['TTM2'] > 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()
    T = (df.CALL_PRICE[ (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()

    X = (df.CALL_PRICE[ (df['TTM2'] < 60) & (df['Moneyness'] >= 1.05)]).count()
    W = (df.CALL_PRICE[ (df['TTM2'] > 60) & (df['TTM2'] < 180) & (df['Moneyness'] >= 1.05)]).count()
    Y = (df.CALL_PRICE[ (df['TTM2'] > 180) & (df['Moneyness'] >= 1.05)]).count()
    Z = (df.CALL_PRICE[ (df['Moneyness'] >= 1.05)]).count()

    rows = [['Days to expiration', '<60', '\se 180', 'All options'],
            ['OTM(<0.95)',     A, B, C, D],
            ['ATM(0.95-1.05)', E, F, G, H],
            ['ITM(>1.05)',     I, J, K, L],
            ['OTM(<0.95)',     M, N, O, P],
            ['ATM(0.95-1.05)', Q, R, S, T],
            ['ITM(>1.05)',     X, W, Y, Z]]

    #print('Tabulate Table:')
    print(tabulate(rows, headers='firstrow'))
    