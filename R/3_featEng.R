dt_train_raw = fread("../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce_train.csv")
dt_test_raw = fread("../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce_test.csv")
# dt_all = fread("../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce_fromPython.csv")

# targetMean --------------------------------------------------------------

for(col in cols_cat){
  
  dt_targetMean_col = dt_train_raw[, c(col, "y"), with = F]
  dt_targetMean_col = dt_targetMean_col[, .(TargetMean = mean(y, na.rm = T)), by = col]
  setnames(dt_targetMean_col, names(dt_targetMean_col), c(col, paste0("Encode_TargetMean_", col)))
  
  dt_train_raw = merge(dt_train_raw, dt_targetMean_col, by = col, all.x = T)
  dt_test_raw = merge(dt_test_raw, dt_targetMean_col, by = col, all.x = T)
  
  # impute
  set(dt_test_raw
      , which(is.na(dt_test_raw[[paste0("Encode_TargetMean_", col)]]))
      , paste0("Encode_TargetMean_", col)
      , mean(dt_train_raw[, y]))
  
}

dim(dt_train_raw); dim(dt_test_raw)


# outlier 1770 -------------------------------------------------------------

# dt_outlier = dt_all[ID == 1770]
# 
# ################
# # original cat #
# ################
# cols = cols_cat
# 
# # exact match
# for(col in cols){
#   
#   dt_all[, paste0("Outlier_1770_Match_", col) := ifelse(get(col) == dt_outlier[[col]], 1, 0)]
#   
# }
# 
# # sum of matches
# cols_toSum = paste0("Outlier_1770_Match_", cols)
# dt_all$Outlier_1770_Match_Cat_Sum = rowSums(dt_all[, cols_toSum, with = F])
# 
# ###############
# # labeled cat #
# ###############
# cols = names(dt_all)[grepl("Encode_Label", names(dt_all))]
# 
# # distant match
# for(col in cols){
#   
#   dt_all[, paste0("Outlier_1770_Distant_", col) := get(col) - dt_outlier[[col]]]
#   dt_all[, paste0("Outlier_1770_Abs_Distant_", col) := abs(get(col) - dt_outlier[[col]])]
#   
# }
# 
# # sum of matches
# cols_toSum = paste0("Outlier_1770_Distant_", cols)
# dt_all$Outlier_1770_Distant_Cat_Sum = rowSums(dt_all[, cols_toSum, with = F])
# cols_toSum = paste0("Outlier_1770_Abs_Distant_", cols)
# dt_all$Outlier_1770_Abs_Distant_Cat_Sum = rowSums(dt_all[, cols_toSum, with = F])
# 
# ##################
# # targetMean cat #
# ##################
# cols = names(dt_all)[grepl("Encode_TargetMean", names(dt_all))]
# 
# # distant match
# for(col in cols){
#   
#   dt_all[, paste0("Outlier_1770_TargetMean_Distant_", col) := get(col) - dt_outlier[[col]]]
#   
# }
# 
# # sum of matches
# cols_toSum = paste0("Outlier_1770_TargetMean_Distant_", cols)
# dt_all$Outlier_1770_TargetMean_Distant_Cat_Sum = rowSums(dt_all[, cols_toSum, with = F])
# 
# ################
# # original bin #
# ################
# cols = cols_bin
# 
# # sum of matches
# sum_match = rep(0, nrow(dt_all))
# for(col in cols){
#   
#   is_match = dt_all[, ifelse(get(col) == dt_outlier[[col]], 1, 0)]
#   sum_match = sum_match + is_match
# }
# 
# dt_all$Outlier_1770_Match_Bin_Sum = sum_match
# 
# ###############################################
# # X314, X29, X265, X47, X261, X54, X118, X315 #
# ###############################################
# cols = c("X314", "X29", "X265", "X47", "X261", "X54", "X118", "X315")
# 
# # exact match
# for(col in cols){
#   
#   dt_all[, paste0("Outlier_1770_Match_", col) := ifelse(get(col) == dt_outlier[[col]], 1, 0)]
#   
# }
# 
# # sum of matches
# cols_toSum = paste0("Outlier_1770_Match_", cols)
# dt_all$Outlier_1770_Match_Important_Bin_Sum = rowSums(dt_all[, cols_toSum, with = F])



# outliers ----------------------------------------------------------------


