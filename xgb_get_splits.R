xgb.get.splits <- function(model, features, min.quality = 0, max.iter = 1000, parallel = 2) {
  require(doParallel)
  cl <- makeCluster(parallel)
  registerDoParallel(cl)
  xgb.tree <- xgb.model.dt.tree(feature_names = features, model = model)
  xgb.tree$Split <- as.numeric(xgb.tree$Split)
  result <- foreach(i = 0:(parallel-1), .combine = function(x,y) rbindlist(list(x,y))) %dopar% ({
    require(stringr)
    xgb.traverse.tree <- function(dt.tree, tree.num, dt.result, max.depth = -1) {
      tree <- tree.num
      curNode <- paste0(tree, "-0")
      curTrav <- 0
      queue <- c(curNode)
      while (TRUE) {
        curNode <- queue[1]
        queue <- queue[-1]
        dt.node <- dt.tree[ID == curNode, ]
        if (dt.node$Feature == "Leaf") {
          break
        }
        if (dt.node$Quality > min.quality) {
          # check to split
          resFeatName <- paste0(dt.node$Feature, " < ", dt.node$Split)
          if (nrow(dt.result[feature == resFeatName]) == 0) {
            dt.result <- rbind(dt.result, data.table(feature = resFeatName, cover = dt.node$Cover,
                                                     quality = dt.node$Quality))
          } else {
            dt.result[feature == resFeatName, cover := cover + dt.node$Cover]
            dt.result[feature == resFeatName, quality := quality + dt.node$Quality]
          }
        }

        # add to queue...
        queue <- c(dt.node$Yes, queue)
        queue <- c(dt.node$No, queue)

        curTrav <- curTrav + 1
        if (curTrav > max.iter) {
          break
        }
      }
      return (dt.result)
    }

    # grab a copy of the template
    dt.features <- data.table(feature = character(0), cover = numeric(0), quality = numeric(0))
    setkey(dt.features, feature)
    for (j in seq(i, max(xgb.tree$Tree), parallel)) {
      dt.features <- xgb.traverse.tree(xgb.tree, j, dt.features, max.depth = max.depth)
    }
    dt.features
  })

  stopCluster(cl)
  # combine the variable importance together
  result <- result[, .(cover = sum(cover), quality = sum(quality)), by = feature]
  setorder(result, -quality)
  return (result)
}

xgb.create.split.features <- function(data, instructions, N = 20) {
  data.x <- copy(data)
  setDT(data.x)
  instructs <- instructions$feature
  iter <- 0
  for (instruct in instructs) {
    i <- str_split(instruct, '<')[[1]]
    featName <- paste0("FEX_", trimws(i[1]), "_", trimws(i[2]))
    data.x[, paste0(featName) := ifelse(get((trimws(i[1]))) < as.numeric(trimws(i[2])), 1, 0) ]
    iter <- iter + 1
    if (iter >= N) {
      break
    }
  }
  return (data.x)
}




##### Example here..
winequality <- read.csv("/proj/analytics/common/Common Data/sampleData/winequality-red.csv", sep = ";")
winequality$quality <- ifelse(winequality$quality > 6, 1, 0)
train.idx <- sample(1:nrow(winequality), 0.8*nrow(winequality), replace=F)
train <- winequality[train.idx,]
eval <- winequality[-train.idx,]
glm.fit <- glm(quality ~ ., data = train, family = "binomial")
glm.pred <- predict(glm.fit, eval)
binaryClassifierEvaluation(glm.pred, eval$quality) # 86.8 AUC

# build a simple xgboost model
xgb.model <- buildModel(data = train, response = "quality") # 91.3 AUC

# extract the splits that xgboost thought was important..
splits <- xgb.get.splits(model = xgb.model$optResult$Models[[xgb.model$optResult$Best_Round$Round]],
                         features = colnames(winequality)[colnames(winequality) != "quality"])

# now embed them into the linear model as 0/1 variables
# control N to get the number of split features..
data.mod <- xgb.create.split.features(data = winequality, instructions = splits, N = 20)
colnames(data.mod)

# run again for logistic regression
train.mod <- data.mod[train.idx,]
eval.mod <- data.mod[-train.idx,]
glm.fit <- glm(quality ~ ., data = train.mod, family = "binomial")
glm.pred <- predict(glm.fit, eval.mod)
binaryClassifierEvaluation(glm.pred, eval.mod$quality) # 89.0 AUC (on 20 features), 90.48 AUC (on 50 features)
