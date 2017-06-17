require(xgboost)
require(MLmetrics)

ls_records = list()
for(i in 1){
  
  # select features ---------------------------------------------------------
  
  cols_featSets = c("Encode_Label"
                    , "PCA", "ICA", "SVD", "FA", "GRP", "SRP")
  
  n_featSets = sample(1:length(cols_featSets), 1)
  cols_featSets_sample_toDrop = sample(cols_featSets, n_featSets)
  print(cols_featSets_sample_toDrop)
  regex_cols_featSets_toDrop = paste0(cols_featSets_sample_toDrop, collapse = "|")
  
  cols_sample = names(dt_all)[!grepl(regex_cols_featSets_toDrop, names(dt_all))]
  
  dt_featSelected = dt_all[, cols_sample, with = F]
  
  # metrics -----------------------------------------------------------------
  
  xg_R_squared = function (yhat, dtrain) {
    
    y = getinfo(dtrain, "label")
    err = R2_Score(yhat, y)
    
    return (list(metric = "error", value = err))
  }
  
  
  # X, y --------------------------------------------------------------------
  
  X_train = dt_featSelected[, !c("y", cols_cat), with = F][ID %in% ids_train]
  X_test = dt_featSelected[, !c("y", cols_cat), with = F][ID %in% ids_test]
  y_train = dt_featSelected[ID %in% ids_train, y]
  
  
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
  
  score_folds = c()
  for(i in 1:1){
    
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
    
    score_fold = cv_xgb$evaluation_log$test_error_mean[cv_xgb$best_iteration]
    print(paste0(i, " - score_fold: ", score_fold))
    
    score_folds = c(score_folds, score_fold)
  }
  
  score_mean = mean(score_folds)
  score_sd = sd(score_folds)
  
  print(paste0("average score: ", score_mean))
  
  # record ------------------------------------------------------------------
  
  ls_records[[i]] = list(cols_featSets_sample_toDrop = cols_featSets_sample_toDrop
                         , score_mean = score_mean
                         , score_sd = score_sd)
  
}



















