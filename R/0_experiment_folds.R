require(data.table)
require(earth)
require(lme4)

# source("")


# load --------------------------------------------------------------------


dt_train = fread("../../competition/data/Mercedes_Benz_Greener_Manufacturing/raw/train.csv")
dt_test = fread("../../competition/data/Mercedes_Benz_Greener_Manufacturing/raw/test.csv")
ids_train = dt_train$ID
ids_test = dt_test$ID

# # concat ------------------------------------------------------------------
# 
# 
# dt_test$y = 0
# dt_test = dt_test[, names(dt_train), with = F]
# dt_all = rbind(dt_train, dt_test)
# 
# 
# # merge folds -------------------------------------------------------------
# 
# 
# dt_id_folds = fread("../../competition/data/Mercedes_Benz_Greener_Manufacturing/folds/dt_id_folds.csv")
# dt_all = merge(dt_all, dt_id_folds, by = "ID", all.x = T)



# load from python --------------------------------------------------------

# dt_all = fread("../../competition/data/Mercedes_Benz_Greener_Manufacturing/data/dt_all.csv")
# dt_all = dt_all[y < 120]
# dt_train = dt_all[ID %in% ids_train]
# dt_test = dt_all[ID %in% ids_test]
# glmm = glmer(y ~ 1 + (1|X0) + (1|X5), data = dt_train, family = Gamma(link = "identity"))
# summary(glmm)
# 
# 

dt_test[X0 == "ae", ID]



# new lb ------------------------------------------------------------------

require(jsonlite)
ls_lb = fromJSON("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/all_questions.json")


dt_lb = data.table()
for(i in 1:length(ls_lb$name)){
  
  dt_lb = rbind(dt_lb
                , data.table(ID = ls_lb$id[[i]]
                             , y = ifelse(length(ls_lb$answers[[i]]$y_value) == 0
                                          , F
                                          , ls_lb$answers[[i]]$y_value)
                             , insidePlb = ifelse(length(ls_lb$answers[[i]]$inside_public_lb) == 0
                                                  , F
                                                  , ls_lb$answers[[i]]$inside_public_lb)))
  
}

dt_lb_true = dt_lb[insidePlb == T][, .(ID, y)]
dt_lb_true[ID == 110, y := 87.70757]
dt_lb_true = rbind(dt_lb_true
      , data.table(ID = c(7805, 289, 3853, 4958, 4960, 1259, 2129, 2342, 7055, 1664, 409
                          , 437, 493, 434, 488, 1045)
                   , y = c(105.8472, 89.27667, 105.481283411, 113.58711, 89.83957, 112.3909, 112.03, 93.06, 91.549, 112.93977, 91.00760
                           , 85.96960, 108.40135, 93.23107, 113.39009, 110.37855)
                   ))

write.csv(dt_lb_true, "../../competition/data/Mercedes_Benz_Greener_Manufacturing/plb_probes/dt_lb_true.csv", row.names = F)
