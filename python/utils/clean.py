import numpy as np

# complimentary cols
def removeCompCols(dt, cols):
    
    seen = []
    col2s = []
    nrow = dt.shape[0]
    for col1 in cols:
        for col2 in cols:
            compliment = np.sum(dt[col1].values + dt[col2].values)
            same = np.sum(dt[col1] == dt[col2])
            if (compliment == nrow) & (same == 0):
                seen.append((col1, col2))
                if (col2, col1) not in seen:
                    col2s.append(col2)
                    print(col1, col2)
    return col2s
