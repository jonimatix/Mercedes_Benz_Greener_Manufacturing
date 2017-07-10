# import
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, "/media/noahhhhhh/dataScience/proj/competition/Mercedes_Benz_Greener_Manufacturing/python/utils/")
import pickle

import numpy as np
import pandas as pd
import gc

# utils
from clean import *
from encode import *
from featureEngineer import *
from model import *
from dups import *


########################### load ###########################
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
dt_train_raw = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/train.csv")
dt_test_raw = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/test.csv")

# ids
ids_train = dt_train_raw.ID.values
ids_test = dt_test_raw.ID.values

# cols
cols_cat = dt_all.drop("ID", axis = 1).select_dtypes(include = ["object"]).columns.values.tolist()
cols_bin = dt_all.filter(regex = "^X").columns.values.tolist()
cols_bin = [x for x in cols_bin if x not in cols_cat]

# ## X. Model

dt_all_train = dt_all[dt_all.ID.isin(ids_train)]
dt_all_test = dt_all[dt_all.ID.isin(ids_test)]

print(dt_all_train.shape, dt_all_test.shape)


# # ### X.1. xgb
# with outlier
#params_cols_dict_best_xgb_withOutlier = featureSelectRun("xgb", dt_all, ids_train, ids_test, cols_cat, cols_bin, rounds = 200)
#pickle.dump(params_cols_dict_best_xgb_withOutlier, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier.pkl", "wb"))

# with outlier and avg duplicates
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
params_cols_dict_best_xgb_withOutlier_aggDupsMedian = featureSelectRun("xgb", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, rounds = 100, autoFolds = True)
pickle.dump(params_cols_dict_best_xgb_withOutlier_aggDupsMedian, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMedian.pkl", "wb"))

dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
params_cols_dict_best_xgb_withOutlier_aggDupsMax = featureSelectRun("xgb", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, rounds = 100, autoFolds = True)
pickle.dump(params_cols_dict_best_xgb_withOutlier_aggDupsMax, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMax.pkl", "wb"))

dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
params_cols_dict_best_xgb_withOutlier_aggDupsMin = featureSelectRun("xgb", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, rounds = 100, autoFolds = True)
params_cols_dict_best_xgb_withOutlier_aggDupsMin = {'dr_grp': False, 'label': False, 'targetMean': True, 'cl_tsne': False, 'dr_fag': False, 'cl_birch': True, 'symbolicTransformer': False, 'dr_ica': False, 'outlierDist': True, 'dr_pca': False, 'targetMeanX5': False, 'dr_tsvd': False, 'ohe': True, 'dr_srp': True, 'dr_nmf': False, 'cl_mds': False, 'targetMeanX0': False}
pickle.dump(params_cols_dict_best_xgb_withOutlier_aggDupsMin, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMin.pkl", "wb"))


# without outlier
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
params_cols_dict_best_xgb_withoutOutlier = featureSelectRun("xgb", dt_all, ids_train, ids_test, cols_cat, cols_bin, rounds = 100, autoFolds = True)
pickle.dump(params_cols_dict_best_xgb_withoutOutlier, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withoutOutlier.pkl", "wb"))

# without outlier and avg duplicates
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
dt_all_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
params_cols_dict_best_xgb_withoutOutlier_aggDupsMedian = featureSelectRun("xgb", dt_all_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, rounds = 100, autoFolds = True)
pickle.dump(params_cols_dict_best_xgb_withoutOutlier_aggDupsMedian, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withoutOutlier_aggDupsMedian.pkl", "wb"))

dt_all.loc[dt_all.y >= 130, "y"] = np.nan
dt_all_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
params_cols_dict_best_xgb_withoutOutlier_aggDupsMax = featureSelectRun("xgb", dt_all_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, rounds = 100, autoFolds = True)
pickle.dump(params_cols_dict_best_xgb_withoutOutlier_aggDupsMax, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withoutOutlier_aggDupsMax.pkl", "wb"))

dt_all.loc[dt_all.y >= 130, "y"] = np.nan
dt_all_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
params_cols_dict_best_xgb_withoutOutlier_aggDupsMin = featureSelectRun("xgb", dt_all_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, rounds = 100, autoFolds = True)
pickle.dump(params_cols_dict_best_xgb_withoutOutlier_aggDupsMin, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withoutOutlier_aggDupsMin.pkl", "wb"))


# # ### X.2. elasticNet
# with outlier
#params_cols_dict_best_elasticNet_withOutlier = featureSelectRun("elasticNet", dt_all, ids_train, ids_test, cols_cat, cols_bin, rounds = 200)
#params_cols_dict_best_elasticNet_withOutlier = {'outlierDist': True, 'dr_nmf': False, 'cl_tsne': True, 'ohe': False, 'dr_tsvd': False, 'dr_srp': False, 'cl_birch': True, 'cl_mds': False, 'targetMeanX0': False, 'label': False, 'dr_pca': False, 'targetMean': False, 'dr_fag': False, 'targetMeanX5': True, 'dr_grp': True, 'dr_ica': True, 'symbolicTransformer': False}
#pickle.dump(params_cols_dict_best_elasticNet_withOutlier, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "wb"))

# without outlier
#dt_all.loc[dt_all.y >= 130, "y"] = np.nan
#params_cols_dict_best_elasticNet_withoutOutlier = featureSelectRun("elasticNet", dt_all, ids_train, ids_test, cols_cat, cols_bin, rounds = 100,  autoFolds = True)
#pickle.dump(params_cols_dict_best_elasticNet_withoutOutlier, open( "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withoutOutlier.pkl", "wb"))


# # ### X.3. knn

# # ### X.4 keras
# with outlier
params_cols_dict_best_keras_withOutlier = featureSelectRun("keras", dt_all, ids_train, ids_test, cols_cat, cols_bin, rounds = 100)
pickle.dump(params_cols_dict_best_keras_withOutlier, open( "../../data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_keras_withOutlier.pkl", "wb"))

## blending
################################################################################## xgb ##################################################################################
# with outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
_, _, dt_preds_xgb_withOutlier = featureSelect("xgb", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withOutlier.csv"
                                   , index = False)

##### with outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMedian.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_xgb_withOutlier_dupsMedian = featureSelect("xgb", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withOutlier_dupsMedian.csv"
                                   , index = False)

##### with outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMax.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_xgb_withOutlier_dupsMax = featureSelect("xgb", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withOutlier_dupsMax.csv"
                                   , index = False)

##### with outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMin.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_xgb_withOutlier_dupsMin = featureSelect("xgb", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withoutOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
_, _, dt_preds_xgb_withoutOutlier = featureSelect("xgb", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withoutOutlier.csv"
                                   , index = False)

##### without outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMedian.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
_, _, dt_preds_xgb_withoutOutlier_dupsMedian = featureSelect("xgb", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withoutOutlier_dupsMedian.csv"
                                   , index = False)

##### without outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMin.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
_, _, dt_preds_xgb_withoutOutlier_dupsMin = featureSelect("xgb", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withoutOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMax.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
_, _, dt_preds_xgb_withoutOutlier_dupsMax = featureSelect("xgb", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_xgb_withoutOutlier_dupsMax.csv"
                                   , index = False)

################################################################################## elasticNet ##################################################################################
# with outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
_, _, dt_preds_elasticNet_withOutlier = featureSelect("elasticNet", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withOutlier.csv"
                                   , index = False)

##### with outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_elasticNet_withOutlier_dupsMedian = featureSelect("elasticNet", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withOutlier_dupsMedian.csv"
                                   , index = False)

##### with outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_elasticNet_withOutlier_dupsMax = featureSelect("elasticNet", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withOutlier_dupsMax.csv"
                                   , index = False)

##### with outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_elasticNet_withOutlier_dupsMin = featureSelect("elasticNet", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
_, _, dt_preds_elasticNet_withoutOutlier = featureSelect("elasticNet", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withoutOutlier.csv"
                                   , index = False)

##### without outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
_, _, dt_preds_elasticNet_withoutOutlier_dupsMedian = featureSelect("elasticNet", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withoutOutlier_dupsMedian.csv"
                                   , index = False)

##### without outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
_, _, dt_preds_elasticNet_withoutOutlier_dupsMin = featureSelect("elasticNet", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withoutOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
_, _, dt_preds_elasticNet_withoutOutlier_dupsMax = featureSelect("elasticNet", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_elasticNet_withoutOutlier_dupsMax.csv"
                                   , index = False)

################################################################################## keras ##################################################################################
# with outlier
params_cols_dict = {'cl_birch': True,
 'cl_mds': False,
 'cl_tsne': True,
 'dr_fag': False,
 'dr_grp': False,
 'dr_ica': True,
 'dr_nmf': True,
 'dr_pca': True,
 'dr_srp': True,
 'dr_tsvd': False,
 'label': False,
 'ohe': True,
 'outlierDist': True,
 'symbolicTransformer': True,
 'targetMean': False,
 'targetMeanX0': True,
 'targetMeanX5': True}
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
_, _, dt_preds_keras_withOutlier = featureSelect("keras", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_keras_withOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_keras_withOutlier.csv"
                                   , index = False)


##### without outlier
params_cols_dict = params_cols_dict = {'cl_birch': True,
 'cl_mds': False,
 'cl_tsne': True,
 'dr_fag': False,
 'dr_grp': False,
 'dr_ica': True,
 'dr_nmf': True,
 'dr_pca': True,
 'dr_srp': True,
 'dr_tsvd': False,
 'label': False,
 'ohe': True,
 'outlierDist': True,
 'symbolicTransformer': True,
 'targetMean': False,
 'targetMeanX0': True,
 'targetMeanX5': True}
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# probes
dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")
for i in dt_lb_true.ID.values:
    dt_all.loc[dt_all.ID == i, "y"] = dt_lb_true[dt_lb_true.ID == i].y.values
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
_, _, dt_preds_keras_withoutOutlier = featureSelect("keras", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_keras_withoutOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/dt_preds_keras_withoutOutlier.csv"
                                   , index = False)


















