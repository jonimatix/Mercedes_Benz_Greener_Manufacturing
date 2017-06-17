# dt_all = fread("../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce.csv")
dt_all = fread("../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce_fromPython.csv")

# targetMean --------------------------------------------------------------

for(col in cols_cat){
  
  dt_targetMean_col = dt_all[ID %in% ids_train][, c(col, "y"), with = F]
  dt_targetMean_col = dt_targetMean_col[, .(TargetMean = mean(y, na.rm = T)), by = col]
  setnames(dt_targetMean_col, names(dt_targetMean_col), c(col, paste0("Encode_TargetMean_", col)))
  
  dt_all = merge(dt_all, dt_targetMean_col, by = col, all.x = T)
  
  # impute
  set(dt_all
      , which(is.na(dt_all[[paste0("Encode_TargetMean_", col)]]))
      , paste0("Encode_TargetMean_", col)
      , mean(dt_all[ID %in% ids_train, y]))
  
}

dim(dt_all)
