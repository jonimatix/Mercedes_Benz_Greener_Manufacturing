require(data.table)

dt_train = fread("../../competition/data/Mercedes_Benz_Greener_Manufacturing/raw/train.csv")
dt_test = fread("../../competition/data/Mercedes_Benz_Greener_Manufacturing/raw/test.csv")
dt_test$y = 0

dt_main = rbind(dt_train[, .(ID, y)], dt_test[, .(ID, y)])

R2 = function(actual, predict){
  
  1 - (sum((actual-predict )^2)/sum((actual-mean(actual))^2))
  
}



# others ------------------------------------------------------------------

directory = "../Kaggle_Benz/blend/"
files = list.files(directory)
files = files[grepl("train", files)]

dt_oof = copy(dt_main)
for(file in files){
  
  dir_file = paste0(directory, file)
  dt_file = fread(dir_file)
  
  name_pred = gsub(".csv|dt_", "", file)
  setnames(dt_file, names(dt_file), c("ID", name_pred, "y"))
  print(nrow(dt_oof))
  
  dt_oof = merge(dt_oof, dt_file[, c("ID", name_pred), with = F], by = "ID")
  
}

# with probes -------------------------------------------------------------

directory = "../data/Mercedes_Benz_Greener_Manufacturing/data/oof/"
files = list.files(directory)[list.files(directory) != "withoutProbes"]
files = files[!grepl("keras", files_noprobs)]
dt_oof = copy(dt_main)
for(file in files){
  
  dir_file = paste0(directory, file)
  dt_file = fread(dir_file)
  
  name_pred = gsub(".csv|dt_", "", file)
  setnames(dt_file, names(dt_file), c("ID", name_pred, "y"))
  print(nrow(dt_oof))
  
  dt_oof = merge(dt_oof, dt_file[, c("ID", name_pred), with = F], by = "ID")
  
}

write.csv(dt_oof, "../data/Mercedes_Benz_Greener_Manufacturing/data/blending/dt_oof.csv", row.names = F)

# glm
model_glmn = glm(y ~ ., data = dt_oof[ID %in% dt_train$ID, !c("ID"), with = F])


# without probes ----------------------------------------------------------

directory_noprobs = "../data/Mercedes_Benz_Greener_Manufacturing/data/oof/withoutProbes/"
files_noprobs = list.files(directory_noprobs)
files_noprobs = files_noprobs[!grepl("keras", files_noprobs)]
dt_oof_noprobs = copy(dt_main)
for(file_noprobs in files_noprobs){
  
  dir_file_noprobs = paste0(directory_noprobs, file_noprobs)
  dt_file_noprobs = fread(dir_file_noprobs)
  
  name_pred_noprobs = gsub(".csv|dt_", "", file_noprobs)
  setnames(dt_file_noprobs, names(dt_file_noprobs), c("ID", name_pred_noprobs, "y"))
  print(nrow(dt_oof_noprobs))
  
  dt_oof_noprobs = merge(dt_oof_noprobs, dt_file_noprobs[, c("ID", name_pred_noprobs), with = F], by = "ID")
  
}

write.csv(dt_oof_noprobs, "../data/Mercedes_Benz_Greener_Manufacturing/data/blending/dt_oof_noprobs.csv", row.names = F)

# avg
preds = rowMeans(dt_oof_noprobs[ID %in% dt_train$ID, !c("ID", "y"), with = F])
R2(dt_oof_noprobs[ID %in% dt_train$ID, y], preds)

# intuition
preds = dt_oof_noprobs[ID %in% dt_train$ID, .7 * preds_xgb_withOutlier + .3 * preds_elasticNet_withOutlier]
R2(dt_oof_noprobs[ID %in% dt_train$ID, y], preds)

# glm
dt_train_all_glm = dt_oof_noprobs[ID %in% dt_train$ID, !c("ID"), with = F]
s = sample(1:nrow(dt_train_all_glm), nrow(dt_train_all_glm) * .2)
dt_train_glm = dt_train_all_glm[-s]
dt_valid_glm = dt_train_all_glm[s]
model_glmn = glm(y ~ ., data = dt_train_glm)
preds = predict(model_glmn, dt_train_glm)
R2(dt_train_glm$y, preds)

# submit
preds_test = dt_oof_noprobs[ID %in% dt_test$ID, .5 * preds_xgb_withOutlier + .3 * preds_elasticNet_withOutlier + .2 * preds_keras_withOutlier]
dt_submit = data.table(ID = dt_test$ID, y = preds_test)
