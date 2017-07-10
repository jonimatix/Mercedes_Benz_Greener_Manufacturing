featureSelect = function(dt_all, params_cols){
  
  # cols
  cols_class = sapply(dt_all, class)
  cols_cat = names(cols_class[cols_class == "character"])
  cols_bin = setdiff(names(dt_all)[grepl("^X", names(dt_all))], cols_cat)
  
  cols_encode_ohe = names(dt_all)[grepl("Encode_ohe_", names(dt_all))]
  cols_encode_label = names(dt_all)[grepl("Encode_Label_", names(dt_all))]
  
  cols_dr_tsvd = names(dt_all)[grepl("DR_TSVD", names(dt_all))]
  cols_dr_pca = names(dt_all)[grepl("DR_PCA", names(dt_all))]
  cols_dr_ica = names(dt_all)[grepl("DR_ICA", names(dt_all))]
  cols_dr_grp = names(dt_all)[grepl("DR_GRP", names(dt_all))]
  cols_dr_srp = names(dt_all)[grepl("DR_SRP", names(dt_all))]
  cols_dr_nmf = names(dt_all)[grepl("DR_NMF", names(dt_all))]
  cols_dr_fag = names(dt_all)[grepl("DR_FAG", names(dt_all))]
  
  cols_outlierDist = names(dt_all)[grepl("outlierDist", names(dt_all))]
  
  cols_cl_tsne = names(dt_all)[grepl("CL_TSNE_", names(dt_all))]
  cols_cl_mds = names(dt_all)[grepl("CL_MDS_", names(dt_all))]
  cols_cl_birch = names(dt_all)[grepl("CL_BIRCH_", names(dt_all))]
  cols_cl_dbscan = names(dt_all)[grepl("CL_DBSCAN", names(dt_all))]
  cols_cl_kmeans = names(dt_all)[grepl("CL_Kmeans", names(dt_all))]
  
  cols = c("y", "ID", "Fold", cols_cat, cols_bin)
  
  if(params_cols$ohe == T){
    cols = c(cols, cols_encode_ohe)
  }else if(params_cols$label == T){
    cols = c(cols, cols_encode_label)
  }else if(params_cols$dr_tsvd == T){
    cols = c(cols, cols_dr_tsvd)
  }else if(params_cols$dr_pca == T){
    cols = c(cols, cols_dr_pca)
  }else if(params_cols$dr_ica == T){
    cols = c(cols, cols_dr_ica)
  }else if(params_cols$dr_grp == T){
    cols = c(cols, cols_dr_grp)
  }else if(params_cols$dr_srp == T){
    cols = c(cols, cols_dr_srp)
  }else if(params_cols$dr_nmf == T){
    cols = c(cols, cols_dr_nmf)
  }else if(params_cols$dr_fag == T){
    cols = c(cols, cols_dr_fag)
  }else if(params_cols$outlierDist == T){
    cols = c(cols, cols_outlierDist)
  }else if(params_cols$cl_tsne == T){
    cols = c(cols, cols_cl_tsne)
  }else if(params_cols$cl_mds == T){
    cols = c(cols, cols_cl_mds)
  }else if(params_cols$cl_birch == T){
    cols = c(cols, cols_cl_birch)
  }else if(params_cols$cl_dbscan == T){
    cols = c(cols, cols_cl_dbscan)
  }else if(params_cols$cl_kmeans == T){
    cols = c(cols, cols_cl_kmeans)
  }
  
  dt_featureSelect = dt_all[cols]
  
}