require(xgboost)
require(MLmetrics)


# select features ---------------------------------------------------------

# dt_all = dt_all[, names(dt_all)[!grepl("LPC", names(dt_all))], with = F]


# metrics -----------------------------------------------------------------

xg_R_squared = function (yhat, dtrain) {
  
  y = getinfo(dtrain, "label")
  err = R2_Score(yhat, y)
  
  return (list(metric = "error", value = err))
}


# X, y --------------------------------------------------------------------

X_train = dt_all[, !c("y", cols_cat), with = F][ID %in% ids_train]
X_test = dt_all[, !c("y", cols_cat), with = F][ID %in% ids_test]
y_train = dt_all[ID %in% ids_train, y]


# xgb.DMatrix -------------------------------------------------------------

dmx_train = xgb.DMatrix(as.matrix(X_train), label = y_train)
dmx_test = xgb.DMatrix(as.matrix(X_test))
ids_test = X_test$ID

# params ------------------------------------------------------------------

params_xgb = list(
  # seed = 123
  subsample = 0.9
  , colsample_bytree = 0.9
  , eta = 0.005
  , objective = 'reg:linear'
  , max_depth = 2
  , min_child_weight = 0
  , alpha = 1
  , lamda = 2
  , gamma = 10
  , num_parallel_tree = 1
  , booster = "gbtree"
  , base_score = mean(y_train)
)


# xgb.cv ------------------------------------------------------------------

for(i in 1:5){
  
  set.seed(i * 888)
  cv_xgb = xgb.cv(params_xgb
                  , dmx_train
                  , nrounds = 10000
                  , nfold = 10
                  , early_stopping_rounds = 50
                  , print_every_n = 50
                  , verbose = 1
                  , feval = xg_R_squared
                  , maximize = T)
  
}


# model -------------------------------------------------------------------

model_xgb = xgb.train(params_xgb, dmx_train
                      , nrounds = cv_xgb$best_iteration
                      , feval = xg_R_squared
                      , maximize = T)


# importance --------------------------------------------------------------

xgb.ggplot.importance(xgb.importance(names(X_train), model = model_xgb), top_n = 50)


# submit ------------------------------------------------------------------

preds_y = predict(model_xgb, dmx_test)
dt_submit = data.table(ID = ids_test
                       , y = preds_y)
head(dt_submit)
dim(dt_submit)

# write.csv(dt_submit, "../data/Mercedes_Benz_Greener_Manufacturing/submission/21_R_basic_targetMean_full_tuned_params.csv", row.names = F)
