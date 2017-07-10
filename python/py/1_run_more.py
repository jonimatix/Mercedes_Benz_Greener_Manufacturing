## blending
################################################################################## xgb ##################################################################################
# with outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
_, _, dt_preds_xgb_withOutlier = featureSelect("xgb", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withOutlier.csv"
                                   , index = False)

##### with outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMedian.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_xgb_withOutlier_dupsMedian = featureSelect("xgb", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withOutlier_dupsMedian.csv"
                                   , index = False)

##### with outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMax.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_xgb_withOutlier_dupsMax = featureSelect("xgb", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withOutlier_dupsMax.csv"
                                   , index = False)

##### with outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMin.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_xgb_withOutlier_dupsMin = featureSelect("xgb", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withoutOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
_, _, dt_preds_xgb_withoutOutlier = featureSelect("xgb", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withoutOutlier.csv"
                                   , index = False)

##### without outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMedian.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
_, _, dt_preds_xgb_withoutOutlier_dupsMedian = featureSelect("xgb", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withoutOutlier_dupsMedian.csv"
                                   , index = False)

##### without outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMin.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
_, _, dt_preds_xgb_withoutOutlier_dupsMin = featureSelect("xgb", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withoutOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_xgb_withOutlier_aggDupsMax.pkl", "rb"))[0]
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
_, _, dt_preds_xgb_withoutOutlier_dupsMax = featureSelect("xgb", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_xgb_withoutOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_xgb_withoutOutlier_dupsMax.csv"
                                   , index = False)

################################################################################## elasticNet ##################################################################################
# with outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
_, _, dt_preds_elasticNet_withOutlier = featureSelect("elasticNet", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withOutlier.csv"
                                   , index = False)

##### with outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_elasticNet_withOutlier_dupsMedian = featureSelect("elasticNet", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withOutlier_dupsMedian.csv"
                                   , index = False)

##### with outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_elasticNet_withOutlier_dupsMax = featureSelect("elasticNet", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withOutlier_dupsMax.csv"
                                   , index = False)

##### with outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
# oof
_, _, dt_preds_elasticNet_withOutlier_dupsMin = featureSelect("elasticNet", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
_, _, dt_preds_elasticNet_withoutOutlier = featureSelect("elasticNet", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withoutOutlier.csv"
                                   , index = False)

##### without outlier agg dups median
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMedian = dups("median", dt_all, cols_cat, cols_bin)
_, _, dt_preds_elasticNet_withoutOutlier_dupsMedian = featureSelect("elasticNet", dt_withOutlier_aggDupsMedian, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier_dupsMedian.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withoutOutlier_dupsMedian.csv"
                                   , index = False)

##### without outlier agg dups min
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMin = dups("min", dt_all, cols_cat, cols_bin)
_, _, dt_preds_elasticNet_withoutOutlier_dupsMin = featureSelect("elasticNet", dt_withOutlier_aggDupsMin, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier_dupsMin.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withoutOutlier_dupsMin.csv"
                                   , index = False)

##### without outlier agg dups max
params_cols_dict = pickle.load(open("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/params_cols_dict_best_elasticNet_withOutlier.pkl", "rb"))
# dt_all
dt_all = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
# dups
dt_withOutlier_aggDupsMax = dups("max", dt_all, cols_cat, cols_bin)
_, _, dt_preds_elasticNet_withoutOutlier_dupsMax = featureSelect("elasticNet", dt_withOutlier_aggDupsMax, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_elasticNet_withoutOutlier_dupsMax.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_elasticNet_withoutOutlier_dupsMax.csv"
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
# oof
_, _, dt_preds_keras_withOutlier = featureSelect("keras", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_keras_withOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_keras_withOutlier.csv"
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
# oof
dt_all.loc[dt_all.y >= 130, "y"] = np.nan
_, _, dt_preds_keras_withoutOutlier = featureSelect("keras", dt_all, ids_train, ids_test, cols_cat, cols_bin, params_cols_dict, autoFolds = True)
dt_preds_keras_withoutOutlier.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/dt_preds_keras_withoutOutlier.csv"
                                   , index = False)