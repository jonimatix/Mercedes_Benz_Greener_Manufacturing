cols_cat = names(dt_all)[sapply(dt_all, is.character)]
cols_bin = names(dt_all)[sapply(dt_all, is.integer)]
cols_bin = cols_bin[!cols_bin %in% "ID"]

# ordered label -----------------------------------------------------------

for(col in cols_cat){
  values = unique(dt_all[[col]])
  values_sorted = sort(values)
  values_sorted_final = values_sorted[order(nchar(values_sorted))]
  
  dt_dict = data.table(col = values_sorted_final
                       , encode = 1:length(values_sorted_final))
  setnames(dt_dict, names(dt_dict), c(col, paste0("Encode_Label_", col)))
  
  dt_all = merge(dt_all, dt_dict, by = col)
  
}

dim(dt_all)


write.csv(dt_all, "../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_encode.csv", row.names = F)
