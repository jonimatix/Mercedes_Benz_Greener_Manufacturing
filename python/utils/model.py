import sys
sys.path.insert(0, "/media/noahhhhhh/dataScience/proj/competition/Mercedes_Benz_Greener_Manufacturing/python/utils/")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

from encode import *
from featureEngineer import *
from model_nn import *

import xgboost as xgb
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import gc

# r_2 for xgboost
def r_2(preds, dtrain):
    labels = dtrain.get_label()
    return 'score', r2_score(labels, preds)
    
# modelling
def model_noah(model, dt_all, ids_train, ids_test, cols_cat, cols_bin, targetMean = True, targetMeanX0 = True, targetMeanX5 = True
, symbolicTransformer = True, verbose = True, autoFolds = False):
    
    if autoFolds == False:
        dt_all_train = dt_all[dt_all["ID"].isin(ids_train)]
        dt_all_test = dt_all[dt_all["ID"].isin(ids_test)]
    else:
        dt_all_train = dt_all[dt_all["y"].notnull()]
        dt_all_test = dt_all[~dt_all["y"].notnull()]
        
        # folds
        k = 10
        bin_y = pd.qcut(dt_all_train.y.values, k, labels = [i for i in range(1, k + 1)]).astype("int64")
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 888)
        dt_id_folds = pd.DataFrame()
        for i, (ind_train, ind_valid) in enumerate(skf.split(dt_all_train, bin_y)):
            dt_id_folds = pd.concat([dt_id_folds
                                    , pd.DataFrame({"ID": dt_all_train.iloc[ind_valid].ID.values
                                                   , "Fold": i + 1})])
        dt_all_train = pd.merge(dt_all_train.drop("Fold", axis = 1), dt_id_folds, on = "ID")

    dt_oof = pd.DataFrame()
    preds_test_fold = np.zeros(dt_all_test.shape[0])
    for f in np.unique(dt_all_train.Fold.values):
        dt_train_fold = dt_all_train[dt_all_train["Fold"] != f]
        dt_valid_fold = dt_all_train[dt_all_train["Fold"] == f]
        dt_test_fold = dt_all_test
        
        
        # targetMean
        if targetMean == True:
            dt_train_fold, dt_valid_fold =  getTargetMean(dt_train_fold, dt_valid_fold, cols_cat, k = 1, random_state = 888)
            _, dt_test_fold =  getTargetMean(dt_train_fold, dt_test_fold, cols_cat, k = 1, random_state = 888)
        else:
            if targetMeanX0 == True:
                dt_train_fold, dt_valid_fold =  getTargetMean(dt_train_fold, dt_valid_fold, ["X0"], k = 1, random_state = 888)
                _, dt_test_fold =  getTargetMean(dt_train_fold, dt_test_fold, ["X0"], k = 1, random_state = 888)
            if targetMeanX5 == True:
                dt_train_fold, dt_valid_fold =  getTargetMean(dt_train_fold, dt_valid_fold, ["X5"], k = 1, random_state = 888)
                _, dt_test_fold =  getTargetMean(dt_train_fold, dt_test_fold, ["X5"], k = 1, random_state = 888)
        
        X_train_fold = dt_train_fold.drop(["Fold", "y"] + cols_cat, axis = 1)
        X_valid_fold = dt_valid_fold.drop(["Fold", "y"] + cols_cat, axis = 1)
        X_test_fold = dt_test_fold.drop(["Fold", "y"] + cols_cat, axis = 1)

        y_train_fold, y_valid_fold = dt_train_fold.y.values, dt_valid_fold.y.values
        
        # symbolicTransformer
        if symbolicTransformer == True:
            X_train_fold_cp = X_train_fold.copy()
            X_train_fold, X_valid_fold = getSymbolTrans(X_train_fold_cp, X_valid_fold, y_train_fold)
            _, X_test_fold = getSymbolTrans(X_train_fold_cp, X_test_fold, y_train_fold)
        
        
        ######################## model ################################
        # xgboost
        if "xgb" in model:
            # xgb.DMatrix
            dmx_train_fold = xgb.DMatrix(X_train_fold, label = y_train_fold)
            dmx_valid_fold = xgb.DMatrix(X_valid_fold, label = y_valid_fold)
            dmx_test_fold = xgb.DMatrix(X_test_fold)
            
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
            preds_test_fold += np.array(model_xgb.predict(dmx_test_fold))
            
        # elasticNet    
        elif "elasticNet" in model:
            # model
            model_cv_elasticNet = ElasticNetCV(normalize = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1], cv = 10, n_jobs = 7)
            model_elasticNet = model_cv_elasticNet.fit(X_train_fold, y_train_fold)
            
            # predict
            preds_valid_fold = model_elasticNet.predict(X_valid_fold)
            preds_test_fold += np.array(model_elasticNet.predict(X_test_fold))
            
        # svr
        elif "svr" in model:
            # model
            model_svr = SVR(kernel = "rbf", C = 1.0, epsilon = 0.05)
            # normalise
            X_all_fold_norm = MinMaxScaler().fit(pd.concat([X_train_fold, X_valid_fold, X_test_fold]))
            X_train_fold_norm = X_all_fold_norm.transform(X_train_fold)
            X_valid_fold_norm = X_all_fold_norm.transform(X_valid_fold)
            X_test_fold_norm = X_all_fold_norm.transform(X_test_fold)
            # fit
            model_svr.fit(X_train_fold_norm, y_train_fold)
            # predict
            preds_valid_fold = model_svr.predict(X_valid_fold_norm)
            preds_test_fold += np.array(model_svr.predict(X_test_fold_norm))
            
        # knn
        elif "knn" in model:
            # features
            cols_tailoredBin = ['X236', 'X127', 'X267', 'X261', 'X383', 'X275', 'X311', 'X189', 'X328',
            'X104', 'X240', 'X152', 'X265', 'X276', 'X162', 'X238', 'X52', 'X117', 'X342',
            'X264', 'X316', 'X339', 'X312', 'X244', 'X77', 'X340', 'X115', 'X38', 'X341',
            'X206', 'X75', 'X203', 'X292', 'X65', 'X221', 'X151', 'X345', 'X198', 'X73',
            'X327', 'X113', 'X196', 'X310']
            
            # normalise
            X_all_fold_norm = MinMaxScaler().fit(pd.concat([X_train_fold, X_valid_fold, X_test_fold]))
            X_train_fold_norm = X_all_fold_norm.transform(X_train_fold[cols_tailoredBin])
            X_valid_fold_norm = X_all_fold_norm.transform(X_valid_fold[cols_tailoredBin])
            X_test_fold_norm = X_all_fold_norm.transform(X_test_fold[cols_tailoredBin])

            # model
            model_knn = KNeighborsRegressor(n_neighbors = 100, weights = "uniform", p = 2)
            model_knn.fit(X_train_fold_norm, y_train_fold)
            
            # predict
            preds_valid_fold = model_knn.predict(X_valid_fold_norm)
            preds_test_fold += np.array(model_xgb.predict(X_test_fold_norm))
            
        # if keras
        elif "keras" in model:
            X_all_fold_norm = MinMaxScaler().fit(pd.concat([X_train_fold, X_valid_fold, X_test_fold]))
            X_train_fold_norm = X_all_fold_norm.transform(X_train_fold)
            X_valid_fold_norm = X_all_fold_norm.transform(X_valid_fold)
            X_test_fold_norm = X_all_fold_norm.transform(X_test_fold)
            
            preds_fold = model_nn_estimator(X_train_fold_norm, y_train_fold, X_valid_fold_norm, y_valid_fold, X_test_fold_norm)
            preds_valid_fold = preds_fold[0]
            preds_test_fold = np.array(preds_fold[1])

        ######################## model ################################
        
        # score
        score_valid = r2_score(y_valid_fold, preds_valid_fold)
        if verbose == True:
            print('Fold %d: Score %f'%(f, score_valid))
            
        # oof prediction
        dt_oof = pd.concat([dt_oof, pd.DataFrame({"ID": dt_valid_fold.ID.values
                                                 , "y": y_valid_fold
                                                 , "preds": preds_valid_fold})])
    
    # oof prediction test
    dt_oof_test = pd.DataFrame({"ID": dt_test_fold.ID.values
                                , "y": 0.0
                                , "preds": preds_test_fold / len(np.unique(dt_all_train.Fold.values))})
          
    # final score
    score_oof = r2_score(dt_oof.y.values, dt_oof.preds)
    print('Final Score %f'%(score_oof))
    
    # concat oof
    dt_oof = pd.concat([dt_oof, dt_oof_test])  
    
    return(dt_oof, score_oof)



# featureSelect
def featureSelect(model, dt, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, verbose = False, autoFolds = False):
    cols_encode_ohe = dt.filter(regex = "Encode_ohe_").columns.values.tolist()
    cols_encode_label = dt.filter(regex = "Encode_Label_").columns.values.tolist()
    
    cols_dr_tsvd = dt.filter(regex = "DR_TSVD").columns.values.tolist()
    cols_dr_pca = dt.filter(regex = "DR_PCA").columns.values.tolist()
    cols_dr_ica = dt.filter(regex = "DR_ICA").columns.values.tolist()
    cols_dr_grp = dt.filter(regex = "DR_GRP").columns.values.tolist()
    cols_dr_srp = dt.filter(regex = "DR_SRP").columns.values.tolist()
    cols_dr_nmf = dt.filter(regex = "DR_NMF").columns.values.tolist()
    cols_dr_fag = dt.filter(regex = "DR_FAG").columns.values.tolist()
    
    cols_outlierDist = dt.filter(regex = "outlierDist").columns.values.tolist()

    cols_cl_tsne = dt.filter(regex = "CL_TSNE_").columns.values.tolist()
    cols_cl_mds = dt.filter(regex = "CL_MDS_").columns.values.tolist()
    cols_cl_birch = dt.filter(regex = "CL_BIRCH_").columns.values.tolist()
    cols_cl_dbscan = dt.filter(regex = "CL_DBSCAN").columns.values.tolist()
    cols_cl_kmeans = dt.filter(regex = "CL_Kmeans").columns.values.tolist()
    
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
    if params_cols_dict["outlierDist"] == True:
        cols = cols + cols_outlierDist
    if params_cols_dict["cl_tsne"] == True:
        cols = cols + cols_cl_tsne + cols_cl_dbscan + cols_cl_kmeans
    if params_cols_dict["cl_mds"] == True:
        cols = cols + cols_cl_mds
    if params_cols_dict["cl_birch"] == True:
        cols = cols + cols_cl_birch

    
#    pdb.set_trace()
   
    dt_featureSelect = dt[cols] 
    
    dt_oof, score = model_noah(model, dt_featureSelect, ids_train, ids_test, cols_cat, cols_bin
    , params_cols_dict["targetMean"], params_cols_dict["targetMeanX0"], params_cols_dict["targetMeanX5"]
    , params_cols_dict["symbolicTransformer"], verbose = verbose, autoFolds = autoFolds)

        
    print(score)
        
    
    return(params_cols_dict, score, dt_oof)

# featureSelectRun
def featureSelectRun(model, dt_all, ids_train, ids_test, cols_cat, cols_bin, rounds = 1, verbose = False, autoFolds = False,):
    res_all = []
    for i in range(1, rounds + 1):
        print(i)
        # params
        params_cols = ["ohe", "label"
                       , "dr_tsvd", "dr_pca", "dr_ica", "dr_grp", "dr_srp", "dr_nmf", "dr_fag", "outlierDist"
                       , "cl_tsne", "cl_mds", "cl_birch"
                       , "targetMean", "targetMeanX0", "targetMeanX5", "symbolicTransformer"]
        values_cols = np.random.choice([True, False], len(params_cols))
        params_cols_dict = dict(zip(params_cols, values_cols))
        print(params_cols_dict)
        
        # feat select
        res = featureSelect(model, dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, verbose, autoFolds)
        res_all.append(res)
        
    params_cols_dict_best_xgb = max(res_all, key = lambda item: item[1])[:2]
    
    gc.collect()
    
    return(params_cols_dict_best_xgb)







