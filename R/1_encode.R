cols_cat = names(dt_train_raw)[sapply(dt_train_raw, is.character)]
cols_bin = names(dt_train_raw)[sapply(dt_train_raw, is.integer)]
cols_bin = cols_bin[!cols_bin %in% "ID"]

# ordered label -----------------------------------------------------------

for(col in cols_cat){
  
  values = unique(c(dt_train_raw[[col]], dt_test_raw[[col]]))
  values_sorted = sort(values)
  values_sorted_final = values_sorted[order(nchar(values_sorted))]
  
  dt_dict = data.table(col = values_sorted_final
                       , encode = 1:length(values_sorted_final))
  setnames(dt_dict, names(dt_dict), c(col, paste0("Encode_Label_", col)))
  
  dt_train_raw = merge(dt_train_raw, dt_dict, by = col)
  dt_test_raw = merge(dt_test_raw, dt_dict, by = col)
  
}

dim(dt_train_raw); dim(dt_test_raw)


write.csv(dt_train_raw, "../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_train_encode.csv", row.names = F)
write.csv(dt_test_raw, "../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_test_encode.csv", row.names = F)
