dt_submit_a = fread("../data/Mercedes_Benz_Greener_Manufacturing/submission/21_R_basic_targetMean_full_tuned_params.csv")
dt_submit_a
# dt_submit_b = fread("../data/Mercedes_Benz_Greener_Manufacturing/submission/26_base_R_more_pca_ica.csv")
# dt_submit_b
setorder(dt_submit, ID)

plot(dt_submit_a$y, dt_submit$y)
