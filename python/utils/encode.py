import numpy as np
import pandas as pd

# orderedLabel
def encode_orderedLabel(dt, cols):
    for c in cols:
        x = list(set(dt[c].values)) 
        x.sort()
        x.sort(key = len)
        dt_labelEncode_c = pd.DataFrame({"Encode_Label_" + c: [i for i in range(1, (len(x) + 1))]
                                         , c: x})
    
        dt = pd.merge(dt, dt_labelEncode_c, on = c)
    
    return(dt)

# targetMean
def getTargetMean(dt_train, dt_test, cols, k = 3, random_state = 888):

    if k == 1:
        train_cp = dt_train.copy()
        test_cp = dt_test.copy()
        for c in cols:
            x = train_cp.groupby([c])["y"].mean()
            dt_targetMean_c = pd.DataFrame({c: x.index
                                           , "Encode_TargetMean_" + c: x.values})
            train_cp = pd.merge(dt_targetMean_c, train_cp, on = c)

            test_cp = pd.merge(dt_targetMean_c, test_cp, on = c, how = "right")
            test_cp = test_cp.fillna(np.mean(train_cp.y))

        return train_cp, test_cp
    else:
        for col in cols:
            X_train_fold = pd.DataFrame()
            X_test_fold = pd.DataFrame()

            skf = StratifiedKFold(n_splits = k, random_state = random_state)

            for i, (ind_in, ind_out) in enumerate(skf.split(dt_train, dt_train[col].values)):
                X_in, X_out = dt_train.iloc[ind_in], dt_train.iloc[ind_out]
                # targetMean in
                dt_targetMean_fold = pd.DataFrame({col: X_in.groupby([col])["y"].mean().index
                                                  , "Encode_TargetMean_" + col: X_in.groupby([col])["y"].mean()})
                # merge targetMean out
                X_out_fold = pd.merge(X_out, dt_targetMean_fold, on = col, how = "left")
                X_out_fold = X_out_fold.fillna(np.mean(X_in.y))

                # concat X_out_fold
                X_train_fold = pd.concat([X_train_fold, X_out_fold])

                # merge with test
                dt_targetMean_fold = dt_targetMean_fold.rename(columns = {"Encode_TargetMean_" + col: "Encode_TargetMean_fold_" + col + "_" + str(i)})
                if i == 0:
                    X_test_fold = pd.merge(dt_test, dt_targetMean_fold, on = col, how = "left")
                else:
                    X_test_fold = pd.merge(X_test_fold, dt_targetMean_fold, on = col, how = "left")

                # mean for test
                cols_encode_fold = X_test_fold.filter(regex = "Encode_TargetMean_fold_").columns.values
                X_test_fold["Encode_TargetMean_" + col] = X_test_fold[cols_encode_fold].mean(axis = 1)
                X_test_fold = X_test_fold.drop(cols_encode_fold, axis = 1)
                X_test_fold = X_test_fold.fillna(np.mean(X_in.y))
    
    return X_train_fold, X_test_fold 