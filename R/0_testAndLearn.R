dt_submit_bset = fread("../data/Mercedes_Benz_Greener_Manufacturing/submission/20_R_basic_targetMean_full.csv")
dt_submit_bset
setorder(dt_submit, ID)

plot(dt_submit_bset$y, dt_submit$y)
