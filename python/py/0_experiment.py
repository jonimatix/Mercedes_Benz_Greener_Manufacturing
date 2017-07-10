# import
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, "/media/noahhhhhh/dataScience/proj/competition/Mercedes_Benz_Greener_Manufacturing/python/utils/")

import numpy as np
import pandas as pd

# utils
from clean import *
from encode import *
from featureEngineer import *
from model import *



# ## 1. Load

# load data
dt_train_raw = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/train.csv")
dt_test_raw = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/test.csv")

# ids fold
dt_id_folds = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/folds/dt_id_folds.csv")

# params_cols
#params_cols_dict_best_xgb = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb.pkl", "rb"))
#params_cols_dict_best_elasticNet = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet.pkl", "rb"))

print(dt_train_raw.shape, dt_test_raw.shape)


# ## 2. transform

# ids
ids_train = dt_train_raw.ID.values
ids_test = dt_test_raw.ID.values

# concat
dt_all = pd.concat([dt_train_raw, dt_test_raw])
# merge folds
dt_all = pd.merge(dt_all, dt_id_folds, how = "left", on = "ID")

print(dt_all.shape)


# ## 3. Clean

# duplicated cols in dt_all
cols_dup_all_toDrop = dt_all.T.duplicated()[dt_all.T.duplicated() == True].index.values
dt_all = dt_all.drop(cols_dup_all_toDrop, axis = 1)

# cols
cols_bin = dt_all.drop("ID", axis = 1).select_dtypes(include = ["int64"]).columns
cols_bin = cols_bin.tolist()
cols_cat = dt_all.drop("ID", axis = 1).select_dtypes(include = ["object"]).columns
cols_cat = cols_cat.tolist()

# removec complimentary cols
cols_comp = removeCompCols(dt_all, cols_bin)

dt_all = dt_all.drop(cols_comp, axis = 1)

# cols
cols_bin = dt_all.drop("ID", axis = 1).select_dtypes(include = ["int64"]).columns
cols_bin = cols_bin.tolist()

print(dt_all.shape)


# ## 4. Encode

# ### 4.1 OHE

dt_cat_onehot = pd.get_dummies(dt_all[cols_cat])
dict_ohe = {x: "Encode_ohe_" + x for x in dt_cat_onehot.columns.values}
dt_cat_onehot = dt_cat_onehot.rename(columns = dict_ohe)
dt_all = dt_all.join(dt_cat_onehot)

dt_all.shape


# ### 4.2 Ordered Label

dt_all = encode_orderedLabel(dt_all, cols_cat)

dt_all.shape


# ## 5. Feature Engineer

# ### 5.1 DR

n_comp = 12
dt_all = getDR(dt_all, n_comp)

dt_all.shape


# ### 5.2 outlierDist

dt_all = outlierDist(dt_all, ids_train, ids_test, cols_cat, cols_bin)

dt_all.shape

# ### 5.3 clustering

dt_all = getClusters(dt_all, cols_cat)

dt_all.shape

dt_all.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv", index = False)







