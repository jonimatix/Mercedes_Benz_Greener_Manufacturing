# Mercedes_Benz_Greener-Manufacturing
a kaggle competition

## Explore
  - all int cols are 1 or 0
  - some cat cols are not in dt_train, but in dt_test
  - some cols are duplicated
  - some cols have the same value
  - the leaderboard is always a lot higher than local

## Init Ideas
  - Cross Validation Strategy
    - Kfold
    - Stratified Kfold on bin y
    - Group Kfold on all cat cols
