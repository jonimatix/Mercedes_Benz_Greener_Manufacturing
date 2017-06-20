require(xgboost)
source("R/0_utils.R")
require(MLmetrics)


# select features ---------------------------------------------------------

# dt_all = dt_all[, names(dt_all)[!grepl("SVD|ICA|SRP|GRP|PCA|FA|1770_Match_Bin_Sum|1770_Match|1770_Distant", names(dt_all))], with = F]
dt_train_raw = dt_train_raw[, names(dt_train_raw)[!grepl("Encode_TargetMean_", names(dt_train_raw))], with = F]
dim(dt_train_raw)

# metrics -----------------------------------------------------------------

xg_R_squared = function (yhat, dtrain) {
  
  y = getinfo(dtrain, "label")
  err = R2_Score(yhat, y)
  
  return (list(metric = "error", value = err))
}


# X, y --------------------------------------------------------------------

X_train = dt_train_raw[, !c("y", cols_cat), with = F]
y_train = dt_train_raw[, y]

X_test = dt_test_raw[, !cols_cat, with = F]


# xgb.DMatrix -------------------------------------------------------------

dmx_train = xgb.DMatrix(as.matrix(X_train), label = y_train)
dmx_test = xgb.DMatrix(as.matrix(X_test))
ids_test = X_test$ID

# params ------------------------------------------------------------------

params_xgb = list(
  seed = 123
  , subsample = 0.85
  , colsample_bytree = 0.8
  , eta = 0.005
  , objective = 'reg:linear'
  , max_depth = 2
  , min_child_weight = 0
  , alpha = 1
  , lamda = 2
  , gamma = 20
  , num_parallel_tree = 1
  , booster = "gbtree"
  , base_score = mean(y_train)
)


# xgb.cv ------------------------------------------------------------------

cv_xgb = xgb.cv(params_xgb
                , dmx_train
                , nrounds = 10000
                , nfold = 10
                , early_stopping_rounds = 50
                , print_every_n = 50
                , verbose = 1
                # , obj = pseudo_huber
                , feval = xg_R_squared
                , maximize = T)
  



# model -------------------------------------------------------------------

vec_preds_y = rep(0, length(ids_test))
n = 10
for(i in 1:n){
  
  cat(paste0(i, " --> "))
  model_xgb = xgb.train(params_xgb, dmx_train
                      , nrounds = cv_xgb$best_iteration
                      , feval = xg_R_squared
                      , maximize = T)
  
  preds_y = predict(model_xgb, dmx_test)
  vec_preds_y = vec_preds_y + preds_y / n
  
}



# importance --------------------------------------------------------------

xgb.ggplot.importance(xgb.importance(names(X_train), model = model_xgb), top_n = 50)


# submit ------------------------------------------------------------------

# preds_y = predict(model_xgb, dmx_test)
dt_submit = data.table(ID = ids_test
                       # , y = preds_y
                       , y = vec_preds_y)
head(dt_submit)
dim(dt_submit)

# write.csv(dt_submit, "../data/Mercedes_Benz_Greener_Manufacturing/submission/29_base_R_no_TargetMean_more_pca_ica_dimensionReduce_separately_correctdly.csv", row.names = F)
