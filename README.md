# Mercedes_Benz_Greener-Manufacturing
a kaggle competition

## Explore
  - all int cols are 1 or 0
  - some cat cols hav values are not in dt_train, but in dt_test
  - some cols are duplicated, some dups in dt_all, some in dt_train but not in dt_test, some the other way around
  - some cols have the same value in dt_train, but not in dt_test, some the other way around, no cols have same value in dt_all
  - the leaderboard is always a lot higher than local (try everything to avoid OVERFIT!!!)

## Init Ideas
  - Cross Validation Strategy (K = 10)
    - Kfold
    - Stratified Kfold on bin y
    - Group Kfold on all cat cols
  - Preprocess
    - drop dup cols in dt_all
      - rename dt_train and dt_test only dup cols 
      - rename dt_train and dt_test only single value cols
    - mark the duplicated rows
    - rename single value cols
    - Encode cat cols
      - OHE
      - TargetMean (oof, 3 fold, remove outlier)
      - Frequency
      - LabelEncoding??? (need investigation)
      - Binary
      - Sum
  - Features
    - mark outlier's cols' value
    - give lower weights to single value cols and duplicates cols that appear in the dt_train
    - use PPtree in R
    - feature interaction
    - PCA, ICA, TSNE, SVG
