require(data.table)


# load --------------------------------------------------------------------


dt_train_raw = fread("../data/Mercedes_Benz_Greener_Manufacturing/raw/train.csv")
dt_test_raw = fread("../data/Mercedes_Benz_Greener_Manufacturing/raw/test.csv")



# transform ---------------------------------------------------------------

dt_test_raw$y = 0
dt_test_raw = dt_test_raw[, names(dt_train_raw), with = F]

# ids
ids_train = dt_train_raw$ID
ids_test = dt_test_raw$ID

# combind
dt_all = rbind(dt_train_raw, dt_test_raw)

dim(dt_all)



# clean -------------------------------------------------------------------

# remove single value
cols_single = c()
for(col in names(dt_train_raw)){
  
  len_col = length(unique(dt_train_raw[[col]]))
  if(len_col == 1){
    cols_single = c(cols_single, col)
  }
  
}

dt_all = dt_all[, !cols_single, with = F]
dim(dt_all)
