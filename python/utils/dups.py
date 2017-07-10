import numpy as np
import pandas as pd
import math

def dups(method, dt_all, cols_cat, cols_bin):

   
    dt_test = dt_all[~dt_all["y"].notnull()]
    dt_train = dt_all[dt_all["y"].notnull()]
    dt_train_dups = dt_train[dt_train[cols_cat + cols_bin].duplicated(keep = False) == True]
    dt_train_nonDups = dt_train[dt_train[cols_cat + cols_bin].duplicated(keep = False) == False]

    if method == "max":
        dt_dups_agg = pd.DataFrame({"y" : dt_train_dups.groupby(cols_cat + cols_bin)["y"].max()}).reset_index()
        dt_nonDups = dt_train_dups[dt_train_dups["y"].isin(dt_dups_agg["y"])]
        dt_nonDup_agg = pd.DataFrame({"ID" : dt_nonDups.groupby(cols_cat + cols_bin)["ID"].max()}).reset_index()
        dt_nonDups_train = dt_train_dups[dt_train_dups["ID"].isin(dt_nonDup_agg["ID"])]
        dt_nonDups_train_others = dt_train_dups[~dt_train_dups["ID"].isin(dt_nonDup_agg["ID"])]
        dt_nonDups_train_others["y"] = np.nan
    elif method == "min":
        dt_dups_agg = pd.DataFrame({"y" : dt_train_dups.groupby(cols_cat + cols_bin)["y"].min()}).reset_index()
        dt_nonDups = dt_train_dups[dt_train_dups["y"].isin(dt_dups_agg["y"])]
        dt_nonDup_agg = pd.DataFrame({"ID" : dt_nonDups.groupby(cols_cat + cols_bin)["ID"].max()}).reset_index()
        dt_nonDups_train = dt_train_dups[dt_train_dups["ID"].isin(dt_nonDup_agg["ID"])]
        dt_nonDups_train_others = dt_train_dups[~dt_train_dups["ID"].isin(dt_nonDup_agg["ID"])]
        dt_nonDups_train_others["y"] = np.nan
    elif method == "median":
        dt_dups_agg = pd.DataFrame({"y" : dt_train_dups.groupby(cols_cat + cols_bin)["y"].apply(lambda x: sorted(x)[math.ceil(len(x) / 2)])}).reset_index()
        dt_nonDups = dt_train_dups[dt_train_dups["y"].isin(dt_dups_agg["y"])]
        dt_nonDup_agg = pd.DataFrame({"ID" : dt_nonDups.groupby(cols_cat + cols_bin)["ID"].max()}).reset_index()
        dt_nonDups_train = dt_train_dups[dt_train_dups["ID"].isin(dt_nonDup_agg["ID"])]
        dt_nonDups_train_others = dt_train_dups[~dt_train_dups["ID"].isin(dt_nonDup_agg["ID"])]
        dt_nonDups_train_others["y"] = np.nan
    
    return(pd.concat([dt_nonDups_train, dt_nonDups_train_others, dt_train_nonDups, dt_test]))