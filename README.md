# Mercedes_Benz_Greener-Manufacturing
a kaggle competition

## Explore
  - all int cols are 1 or 0
  - some cat cols hav values are not in dt_train, but in dt_test
  - some cols are duplicated, some dups in dt_all, some in dt_train but not in dt_test, some the other way around
  - some cols have the same value in dt_train, but not in dt_test, some the other way around, no cols have same value in dt_all
  - the leaderboard is always a lot higher than local (try everything to avoid OVERFIT!!!)

## Init Ideas
  - Cross Validation Strategy
    - Kfold
    - Stratified Kfold on bin y (k = 8)
    - Group Kfold on all cat cols
  - Clean
    - drop dup cols in dt_all (Done)
      - rename dt_train and dt_test only dup cols (Done)
      - rename dt_train and dt_test only single value cols (Done)
    - mark the duplicated rows (Done)
      - all cols (Done)
      - int cols (Done)
      - cat cols (Done)
      - mean y of dup rows (what about test?)
    - rename single value cols (Done)
    - remove complimentary cols (Done)
   - Encode
     - OHE (Done)
     - TargetMean (oof, 3 fold, remove outlier) (Done)
     - TargetMedian
     - Frequency (Done)
     - Binary (Done)
     - Ordinal (Done)
  - Features
    - ID (Done)
    - mark outlier's cols' value (Done)
      - for cat cols (Done)
      - for int cols (Done)
      - for all cols (Done)
      - for X0 (Done)
    - sum of binary features (Done)
      - all binary (Done)
      - most important 3, 5, 10, 20, 50 binary cols (Done)
    - mean of y of IDs who are nearest (Done)
    - Dimension Reduction
      - PCA (Done)
      - ICA (Done)
      - SVD (Done)
      - FA (Done)
      - Logistic PCA (havent done)
      - TSNE (Done)
    - give lower weights to single value cols and duplicates cols that appear in the dt_train
    - use PPtree in R
    - feature interaction (XOR)
  - Model Tuning
    - pipline and cv all parameters
      - Dimension Reduction n_component
      - model params
      - feature sets
## Revised, TODO
- Symbolic Transformer (http://gplearn.readthedocs.io/en/latest/examples.html#example-2-symbolic-tranformer)
- try aggregate duplicates (take the max?)
- use train and test, instead of dt_all
- architecture: skf oof
  - features: oof distance to outliers
  - features: oof targetMean inside of skf targetMean 
- multiple models ensemble
