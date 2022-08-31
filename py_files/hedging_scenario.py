import pandas as pd
from scipy import stats
import numpy as np


def hedging_func(df):

    df = df.sort_values(by=["A", "Name"])
    df['count'] = (
        df.groupby(["A"])['Name'].transform('nunique'))
    df = df[df["count"] >= 50]
    df["VS"] = df["UNDERLYING"] * df["deltaANN"]
    # initializing lag parameter
    df["lag_deltaANN"] = df.deltaANN.shift()
    df["rank"] = df.groupby("A").Name.transform('cumcount')
    df["VC"] = -df["bsprice"]
    #df["VB"] = np.where(df["rank"] == 0, -df["VS"] + df["VC"],0)
    df["VB"] = -df["VS"] + df["VC"]
    df["VB"] = df["VB"].astype(float, errors='raise')
    df["VBlag"] = df.VB.shift()
    #df["VB"] = np.where(df["rank"] > 0, (np.exp(df["RF"]) * df["VBlag"]) - df["UNDERLYING"] * (df["deltaANN"] - df["lag_deltaANN"]), df["VB"])
    df["VB"] = np.where(df["rank"] > 0, (np.exp(0.01) * df["VBlag"]) -
                        df["UNDERLYING"] * (df["deltaANN"] - df["lag_deltaANN"]), df["VB"])

    df["VT"] = df["VS"] + df["VB"] + df["VC"]

    df["VSBS"] = df["UNDERLYING"] * df["bsdelta"]
    df["lag_deltabs"] = df.bsdelta.shift()
    df["VCBS"] = -df["bsprice"]
    df["VBBS"] = -df["VSBS"] + df["VCBS"]
    df["VBBS"] = df["VBBS"].astype(float, errors='raise')
    df["VBBSlag"] = df.VBBS.shift()
    #df["VBBS"] = np.where(df["rank"] > 0, (np.exp(df["RF"]) * df["VBBSlag"]) - df["UNDERLYING"] * (df["deltaANN"] - df["lag_deltaANN"]), df["VB"])
    df["VBBS"] = np.where(df["rank"] > 0, (np.exp(0.01) * df["VBBSlag"]) -
                          df["UNDERLYING"] * (df["deltaANN"] - df["lag_deltaANN"]), df["VB"])
    df["VTBS"] = df["VSBS"] + df["VBBS"] + df["VCBS"]
    grouped_df = df.groupby(['A']).tail(1)

    print(grouped_df["VT"].abs().mean(), "ANN")
    print(grouped_df["VTBS"].abs().mean(), "BS")

    tracking_error_ANN = grouped_df["VT"].abs()
    tracking_error_BS = grouped_df["VTBS"].abs()

    print(stats.ttest_rel(grouped_df["VT"].abs(), grouped_df["VTBS"].abs()))
