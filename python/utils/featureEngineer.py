import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.cluster import FeatureAgglomeration
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer

# dimension reduction
def getDR(dt_all, n_comp = 12):
    # cols
    cols_encode_label = dt_all.filter(regex = "Label").columns.values.tolist()
    cols_cat = dt_all.drop("ID", axis = 1).select_dtypes(include = ["object"]).columns.tolist()
    
    # standardize
    dt_all_norm = MinMaxScaler().fit_transform(dt_all.drop(["y", "Fold"] + cols_cat + cols_encode_label, axis = 1))

    # tSVD
    tsvd = TruncatedSVD(n_components = n_comp, random_state = 420)
    tsvd_results = tsvd.fit_transform(dt_all_norm)
    
    # PCA
    pca = PCA(n_components = n_comp, random_state = 420)
    pca_results = pca.fit_transform(dt_all_norm)
    
    # ICA
    ica = FastICA(n_components = n_comp, max_iter = 5000, random_state = 420)
    ica_results = ica.fit_transform(dt_all_norm)
    
    # GRP
    grp = GaussianRandomProjection(n_components = n_comp, eps = 0.1, random_state = 420)
    grp_results = grp.fit_transform(dt_all_norm)
    
    # SRP
    srp = SparseRandomProjection(n_components = n_comp, dense_output = True, random_state = 420)
    srp_results = srp.fit_transform(dt_all_norm)
    
    # NMF
    nmf = NMF(n_components = n_comp, init = 'nndsvdar', random_state = 420)
    nmf_results = nmf.fit_transform(dt_all_norm)
    
    # FAG
    fag = FeatureAgglomeration(n_clusters = n_comp, linkage = 'ward')
    fag_results = fag.fit_transform(dt_all_norm)
    
    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        dt_all['TSVD_' + str(i)] = tsvd_results[:, i - 1]
        dt_all['PCA_' + str(i)] = pca_results[:, i - 1]
        dt_all['ICA_' + str(i)] = ica_results[:, i - 1]
        dt_all['GRP_' + str(i)] = grp_results[:, i - 1]
        dt_all['SRP_' + str(i)] = srp_results[:, i - 1]
        dt_all['NMF_' + str(i)] = nmf_results[:, i - 1]
        dt_all['FAG_' + str(i)] = fag_results[:, i - 1]
    
    return(dt_all)
    

# symbolic transformation
def getSymbolTrans(X_train, X_valid, y_train, random_state = 888):
    function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
    
    gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=0,
                         random_state=0, n_jobs=3)

    gp.fit(X_train, y_train)
    
    gp_features_train = gp.transform(X_train)
    dt_gp_features_train = pd.DataFrame(gp_features_train)
    dt_gp_features_train.columns = ["ST_" + str(i) for i in range(1, dt_gp_features_train.shape[1] + 1)]
    X_train = X_train.join(dt_gp_features_train)
    X_train = X_train.fillna(0)
    
    gp_features_valid = gp.transform(X_valid)
    dt_gp_features_valid = pd.DataFrame(gp_features_valid)
    dt_gp_features_valid.columns = ["ST_" + str(i) for i in range(1, dt_gp_features_valid.shape[1] + 1)]
    X_valid = X_valid.join(dt_gp_features_valid)
    X_valid = X_valid.fillna(0)
    
    return(X_train, X_valid)