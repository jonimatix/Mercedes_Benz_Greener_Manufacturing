import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score


dt_train_raw = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/train.csv")
dt_test_raw = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/test.csv")

dt_oof = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/blending/dt_oof.csv")
dt_oof_noprobs = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/data/blending/dt_oof_noprobs.csv")

dt_lb_true = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv")

# probes

X = dt_oof.drop(["ID", "y"], axis = 1)
y = dt_oof.y.values

X_train = dt_oof[dt_oof["ID"].isin(dt_train_raw.ID.values)].drop(["ID", "y"], axis = 1)
y_train = dt_oof[dt_oof["ID"].isin(dt_train_raw.ID.values)].y.values

r2_score(y_train, X_train["preds_xgb_withOutlier"].values)


X_test = dt_oof[dt_oof["ID"].isin(dt_test_raw.ID.values)].drop(["ID", "y"], axis = 1)
y_test = dt_oof[dt_oof["ID"].isin(dt_test_raw.ID.values)].y.values

# model
model_cv_elasticNet = ElasticNetCV(normalize = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1], cv = 10, n_jobs = 7)
model_elasticNet = model_cv_elasticNet.fit(X_train, y_train)

# cv score
scores = cross_val_score(model_elasticNet, X_train, y_train, cv = 10, scoring = 'r2')

# pred
preds = model_elasticNet.predict(X_test)

# submit
dt_submit = pd.DataFrame({"ID": dt_test_raw.ID.values
                          , "y": preds})

# check with probes
dt_submit[dt_submit["ID"].isin(dt_lb_true.ID)]

dt_submit.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/submission/blending/01_blend_noah_robes.csv", index = False)

# no probes

X = dt_oof_noprobs.drop(["ID", "y"], axis = 1)
y = dt_oof_noprobs.y.values

X_train = dt_oof_noprobs[dt_oof_noprobs["ID"].isin(dt_train_raw.ID.values)].drop(["ID", "y"], axis = 1)
y_train = dt_oof_noprobs[dt_oof_noprobs["ID"].isin(dt_train_raw.ID.values)].y.values

r2_score(y_train, X_train["preds_xgb_withOutlier"].values)


X_test = dt_oof_noprobs[dt_oof_noprobs["ID"].isin(dt_test_raw.ID.values)].drop(["ID", "y"], axis = 1)
y_test = dt_oof_noprobs[dt_oof_noprobs["ID"].isin(dt_test_raw.ID.values)].y.values

# model
model_cv_elasticNet = ElasticNetCV(normalize = True, l1_ratio = [.1, .5, .7, .9, .95, .99, 1], cv = 10, n_jobs = 7)
model_elasticNet = model_cv_elasticNet.fit(X_train, y_train)

# cv score
scores = cross_val_score(model_elasticNet, X_train, y_train, cv = 10, scoring = 'r2')

# pred
preds = model_elasticNet.predict(X_test)

# submit
dt_submit = pd.DataFrame({"ID": dt_test_raw.ID.values
                          , "y": preds})

# check with probes
dt_submit[dt_submit["ID"].isin(dt_lb_true.ID)]

dt_submit.to_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/submission/blending/01_blend_noah_noprobes.csv", index = False)