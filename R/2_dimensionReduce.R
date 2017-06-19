require(caret)
require(fastICA)
require(logisticPCA)
require(rARPACK)
require(RandPro)

n_comp = 50

# #  pca --------------------------------------------------------------------
# 
# set.seed(888)
# dr_pca = preProcess(dt_all[, !c("y", cols_cat), with = F]
#                     , method = "pca"
#                     , thresh = .95
#                     # , pcaComp = n_comp
#                     )
# dt_pca = predict(dr_pca, dt_all[, !c("y", cols_cat), with = F])
# 
# 
# # logistic pca ------------------------------------------------------------
# 
# # dr_lpca = logisticPCA(dt_all[, cols_bin, with = F], k = n_comp, m = 0, partial_decomp = T)
# # dt_lpca = predict(dr_lpca, dt_all[, cols_bin, with = F], type = "PCs")
# # colnames(dt_lpca) = paste0("LPC_", 1:n_comp)
# 
# # ica ---------------------------------------------------------------------
# 
# 
# dr_ica = fastICA(dt_all[, !c("y", cols_cat), with = F]
#                  , n.comp = n_comp
#                  , alg.typ = "parallel"
#                  , fun = "logcosh"
#                  , method = "C"
#                  , row.norm = T)
# 
# dt_ica = as.data.table(dr_ica$S)
# setnames(dt_ica, names(dt_ica), paste0("ICA_", 1:n_comp))
# 
# 
# # combine -----------------------------------------------------------------
# 
# 
# dt_all = cbind(dt_all, dt_pca); dim(dt_all)
# dt_all = cbind(dt_all, dt_ica); dim(dt_all)
# # dt_all = cbind(dt_all, dt_lpca); dim(dt_all)





############################################
# DO IT SEPARATELY #########################
############################################

#  pca --------------------------------------------------------------------

set.seed(888)
dr_pca = preProcess(dt_train_raw[, !c("y", cols_cat), with = F]
                    , method = "pca"
                    , thresh = .95
                    , uniqueCut = 1
                    # , pcaComp = n_comp
)
dt_pca_train = predict(dr_pca, dt_train_raw[, !c("y", cols_cat), with = F])
dt_pca_test = predict(dr_pca, dt_test_raw[, !c(cols_cat), with = F])


# ica ---------------------------------------------------------------------

dr_ica_train = fastICA(dt_train_raw[, !c("y", cols_cat), with = F]
                 , n.comp = n_comp
                 , alg.typ = "parallel"
                 , fun = "logcosh"
                 , method = "C"
                 , row.norm = T)

dt_ica_train = as.data.table(dr_ica_train$S)
setnames(dt_ica_train, names(dt_ica_train), paste0("ICA_", 1:n_comp))

dr_ica_test = fastICA(dt_test_raw[, !c(cols_cat), with = F]
                       , n.comp = n_comp
                       , alg.typ = "parallel"
                       , fun = "logcosh"
                       , method = "C"
                       , row.norm = T)

dt_ica_test = as.data.table(dr_ica_test$S)
setnames(dt_ica_test, names(dt_ica_test), paste0("ICA_", 1:n_comp))



# combine -----------------------------------------------------------------

# pca
dt_train_raw = cbind(dt_train_raw, dt_pca_train)
dt_test_raw = cbind(dt_test_raw, dt_pca_test)
# ica
dt_train_raw = cbind(dt_train_raw, dt_ica_train)
dt_test_raw = cbind(dt_test_raw, dt_ica_test)

dim(dt_train_raw); dim(dt_test_raw)





## save ##

write.csv(dt_train_raw, "../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce_train.csv", row.names = F)
write.csv(dt_test_raw, "../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce_test.csv", row.names = F)
