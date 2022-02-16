import pandas as pd



from tabulate import tabulate
import numpy as np



def pricesummary(df):

        
    
    A = np.mean(df[ (df['TTM'] < 60) & (df['Moneyness'] < 0.95)])
    B = np.mean(df[ (df['TTM'] > 60) & (df['TTM'] < 180) & (df['Moneyness'] < 0.95)])
    C = np.mean(df[ (df['TTM'] > 180) & (df['Moneyness'] < 0.95)])
    D = np.mean(df[ (df['Moneyness'] < 0.95)])

    E = np.mean(df[ (df['TTM'] < 60) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])
    F = np.mean(df[ (df['TTM'] > 60) & (df['TTM'] < 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])
    G = np.mean(df[ (df['TTM'] > 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])
    H = np.mean(df[ (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)])

    I = np.mean(df[ (df['TTM'] < 60) & (df['Moneyness'] >= 1.05)])
    J = np.mean(df[ (df['TTM'] > 60) & (df['TTM'] < 180) & (df['Moneyness'] >= 1.05)])
    K = np.mean(df[ (df['TTM'] > 180) & (df['Moneyness'] >= 1.05)])
    L = np.mean(df[ (df['Moneyness'] >= 1.05)])

    M = (df[ (df['TTM'] < 60) & (df['Moneyness'] < 0.95)]).count()
    N = (df[ (df['TTM'] > 60) & (df['TTM'] < 180) & (df['Moneyness'] < 0.95)]).count()
    O = (df[ (df['TTM'] > 180) & (df['Moneyness'] < 0.95)]).count()
    P = (df[ (df['Moneyness'] < 0.95)]).count()

    Q = (df[ (df['TTM'] < 60) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()
    R = (df[ (df['TTM'] > 60) & (df['TTM'] < 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()
    S = (df[ (df['TTM'] > 180) & (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()
    T = (df[ (df['Moneyness'] > 0.95) & (df['Moneyness'] < 1.05)]).count()

    X = (df[ (df['TTM'] < 60) & (df['Moneyness'] >= 1.05)]).count()
    W = (df[ (df['TTM'] > 60) & (df['TTM'] < 180) & (df['Moneyness'] >= 1.05)]).count()
    Y = (df[ (df['TTM'] > 180) & (df['Moneyness'] >= 1.05)]).count()
    Z = (df[ (df['Moneyness'] >= 1.05)]).count()

    rows = [['Days to expiration', '<60', '\se 180', 'All options'],
            ['OTM(<0.95)',     A, B, C, D],
            ['ATM(0.95-1.05)', E, F, G, H],
            ['ITM(>1.05)',     I, J, K, L],
            ['OTM(<0.95)',     M, N, O, P],
            ['ATM(0.95-1.05)', Q, R, S, T],
            ['ITM(>1.05)',     X, W, Y, Z]]

    #print('Tabulate Table:')
    print(tabulate(rows, headers='firstrow'))
    