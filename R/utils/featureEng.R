getTargetMean = function(dt_train_raw, dt_test_raw){
  
  for(col in c("X0", "X5")){
    
    
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
  
  return(list(dt_train = dt_train_raw, dt_test = dt_test_raw))
  
}