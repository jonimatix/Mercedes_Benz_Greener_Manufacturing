require(caret)
require(fastICA)
require(logisticPCA)
require(rARPACK)
require(RandPro)

n_comp = 12

#  pca --------------------------------------------------------------------

set.seed(888)
dr_pca = preProcess(dt_all[, !c("y", cols_cat), with = F]
                    , method = "pca"
                    # , thresh = .8
                    , pcaComp = n_comp
                    )
dt_pca = predict(dr_pca, dt_all[, !c("y", cols_cat), with = F])


# logistic pca ------------------------------------------------------------

# dr_lpca = logisticPCA(dt_all[, cols_bin, with = F], k = n_comp, m = 0, partial_decomp = T)
# dt_lpca = predict(dr_lpca, dt_all[, cols_bin, with = F], type = "PCs")
# colnames(dt_lpca) = paste0("LPC_", 1:n_comp)

# ica ---------------------------------------------------------------------


dr_ica = fastICA(dt_all[, !c("y", cols_cat), with = F]
                 , n.comp = n_comp
                 , alg.typ = "parallel"
                 , fun = "logcosh"
                 , method = "C"
                 , row.norm = T)

dt_ica = as.data.table(dr_ica$S)
setnames(dt_ica, names(dt_ica), paste0("ICA_", 1:n_comp))



# pptree ------------------------------------------------------------------

# bin_y = dt_test_raw$y
# 
# PPindex.class()


# combine -----------------------------------------------------------------


dt_all = cbind(dt_all, dt_pca); dim(dt_all)
dt_all = cbind(dt_all, dt_ica); dim(dt_all)
# dt_all = cbind(dt_all, dt_lpca); dim(dt_all)



















## save ##

write.csv(dt_all, "../data/Mercedes_Benz_Greener_Manufacturing/data/R/dt_dimensionReducce.csv", row.names = F)
