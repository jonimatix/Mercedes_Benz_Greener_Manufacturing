import sys
sys.path.insert(0, "/media/noahhhhhh/dataScience/proj/competition/Mercedes_Benz_Greener_Manufacturing/python/utils/")

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from encode import *
from featureEngineer import *

from sklearn.linear_model import ElasticNet, ElasticNetCV
import xgboost as xgb

# r_2 for xgboost
def r_2(preds, dtrain):
    labels = dtrain.get_label()
    return 'score', r2_score(labels, preds)
    
# xgboost
def model_xgb(model, dt_all_train, cols_cat, targetMean = True, symbolicTransformer = True, verbose = True):
    dt_oof = pd.DataFrame()
    for f in np.unique(dt_all_train.Fold.values):
        dt_train_fold = dt_all_train[dt_all_train["Fold"] != f]
        dt_valid_fold = dt_all_train[dt_all_train["Fold"] == f]

        # targetMean
        if targetMean == True:
            dt_train_fold, dt_valid_fold =  getTargetMean(dt_train_fold, dt_valid_fold, cols_cat, k = 1, random_state = 888)

        X_train_fold = dt_train_fold.drop(["Fold", "y"] + cols_cat, axis = 1)
        X_valid_fold = dt_valid_fold.drop(["Fold", "y"] + cols_cat, axis = 1)

        y_train_fold, y_valid_fold = dt_train_fold.y.values, dt_valid_fold.y.values
        
        # symbolicTransformer
        if symbolicTransformer == True:
            X_train_fold, X_valid_fold = getSymbolTrans(X_train_fold, X_valid_fold, y_train_fold)
        
        ######################## model ################################
        # xgboost
        if "xgb" in model:
            # xgb.DMatrix
            dmx_train_fold = xgb.DMatrix(X_train_fold, label = y_train_fold)
            dmx_valid_fold = xgb.DMatrix(X_valid_fold, label = y_valid_fold)
    
            ls_watch =  [(dmx_train_fold, "train"), (dmx_valid_fold, "valid")]
    
            # params
            params_xgb = {
                "objective": "reg:linear"
                , "booster": "gbtree"
                , "learning_rate": 0.005
                , "subsample": .9
                , "colsample": .8
                , "max_depth": 2
                , "alpha": 1
                , "lambda": 2
                , "gamma": 20
                , "base_score": np.mean(y_train_fold)
                , "nthread": 7
            }
    
            # model
            model_xgb = xgb.train(params_xgb, dmx_train_fold, evals = ls_watch
                                  , num_boost_round = 5000
                                  , feval = r_2, maximize = True, early_stopping_rounds = 50
                                  , verbose_eval = False)
            
            # predict
            preds_valid_fold = model_xgb.predict(dmx_valid_fold)
        
        # elasticNet    
        elif "elasticNet" in model:
            # model
            model_cv_elasticNet = ElasticNetCV(normalize = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1], cv = 10, n_jobs = 7)
            model_elasticNet = model_cv_elasticNet.fit(X_train_fold, y_train_fold)
            
            # predict
            preds_valid_fold = model_elasticNet.predict(X_valid_fold)
        
        ######################## model ################################
        
        # score
        score_valid = r2_score(y_valid_fold, preds_valid_fold)
        if verbose == True:
            print('Fold %d: Score %f'%(f, score_valid))
            
        # oof prediction
        dt_oof = pd.concat([dt_oof, pd.DataFrame({"ID": dt_valid_fold.ID.values
                                                 , "y": y_valid_fold
                                                 , "preds": preds_valid_fold})])
    
    # final score
    score_oof = r2_score(dt_oof.y.values, dt_oof.preds)
    print('Final Score %f'%(score_oof))
    
    return(score_oof)


# featureSelect
def featureSelect(model, dt, cols_cat, cols_bin, params_cols_dict):
    cols_encode_ohe = dt.filter(regex = "ohe").columns.values.tolist()
    cols_encode_label = dt.filter(regex = "Label").columns.values.tolist()
    
    cols_dr_tsvd = dt.filter(regex = "TSVD").columns.values.tolist()
    cols_dr_pca = dt.filter(regex = "PCA").columns.values.tolist()
    cols_dr_ica = dt.filter(regex = "ICA").columns.values.tolist()
    cols_dr_grp = dt.filter(regex = "GRP").columns.values.tolist()
    cols_dr_srp = dt.filter(regex = "SRP").columns.values.tolist()
    cols_dr_nmf = dt.filter(regex = "NMF").columns.values.tolist()
    cols_dr_fag = dt.filter(regex = "FAG").columns.values.tolist()

    cols = ["y", "ID", "Fold"] + cols_cat + cols_bin
    if params_cols_dict["ohe"] == True:
        cols = cols + cols_encode_ohe
    if params_cols_dict["label"] == True:
        cols = cols + cols_encode_label
    if params_cols_dict["dr_tsvd"] == True:
        cols = cols + cols_dr_tsvd
    if params_cols_dict["dr_pca"] == True:
        cols = cols + cols_dr_pca
    if params_cols_dict["dr_ica"] == True:
        cols = cols + cols_dr_ica
    if params_cols_dict["dr_grp"] == True:
        cols = cols + cols_dr_grp
    if params_cols_dict["dr_srp"] == True:
        cols = cols + cols_dr_srp
    if params_cols_dict["dr_nmf"] == True:
        cols = cols + cols_dr_nmf
    if params_cols_dict["dr_fag"] == True:
        cols = cols + cols_dr_fag
        
    dt_featureSelect = dt[cols]
    
    score = model_xgb(model, dt_featureSelect, cols_cat, params_cols_dict["targetMean"], params_cols_dict["symbolicTransformer"], verbose = False)

        
    print(score)
        
    list_res = [(params_cols_dict, score)]
    return(list_res)