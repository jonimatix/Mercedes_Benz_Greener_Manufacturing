import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, NMF
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.cluster import FeatureAgglomeration
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import Birch, DBSCAN, KMeans

from scipy.spatial.distance import pdist, squareform, cdist

# dimension reduction
def getDR(dt_all, n_comp = 12):
    # cols
    cols_encode_label = dt_all.filter(regex = "Encode_Label").columns.values.tolist()
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
        dt_all['DR_TSVD_' + str(i)] = tsvd_results[:, i - 1]
        dt_all['DR_PCA_' + str(i)] = pca_results[:, i - 1]
        dt_all['DR_ICA_' + str(i)] = ica_results[:, i - 1]
        dt_all['DR_GRP_' + str(i)] = grp_results[:, i - 1]
        dt_all['DR_SRP_' + str(i)] = srp_results[:, i - 1]
        dt_all['DR_NMF_' + str(i)] = nmf_results[:, i - 1]
        dt_all['DR_FAG_' + str(i)] = fag_results[:, i - 1]
    
    return(dt_all)
    

# symbolic transformation
def getSymbolTrans(train, valid, y, random_state = 888):
    
    X_train = train.copy()
    X_valid = valid.copy()
    y_train = y.copy()
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

# outlierDist
def outlierDist(dt_all, ids_train, ids_test, cols_cat, cols_bin):
    
    cols_encode_ohe = dt_all.filter(regex = "Encode_ohe").columns.values.tolist()
    cols_encode_label = dt_all.filter(regex = "Encode_Label").columns.values.tolist()
    
    cols_remove = dt_all.filter(regex = "DR").columns.values.tolist()
    
    dt_all_train = dt_all[dt_all.ID.isin(ids_train)]
    dt_all_test = dt_all[dt_all.ID.isin(ids_test)]
    cols_tailoredBin = ['X236', 'X127', 'X267', 'X261', 'X383', 'X275', 'X311', 'X189', 'X328',
            'X104', 'X240', 'X152', 'X265', 'X276', 'X162', 'X238', 'X52', 'X117', 'X342',
            'X264', 'X316', 'X339', 'X312', 'X244', 'X77', 'X340', 'X115', 'X38', 'X341',
            'X206', 'X75', 'X203', 'X292', 'X65', 'X221', 'X151', 'X345', 'X198', 'X73',
            'X327', 'X113', 'X196', 'X310']
            
    distance_test_ohe = np.zeros(dt_all_test.shape[0])
    distance_test_label = np.zeros(dt_all_test.shape[0])
    distance_test_bin = np.zeros(dt_all_test.shape[0])
    distance_test_tailoredBin = np.zeros(dt_all_test.shape[0])
    distance_test_all = np.zeros(dt_all_test.shape[0])
    
    dt_train_outlier = pd.DataFrame()
    for f in np.unique(dt_all_train.Fold.values):
        dt_train_fold = dt_all_train[dt_all_train["Fold"] != f]
        dt_valid_fold = dt_all_train[dt_all_train["Fold"] == f]
        
        # max y row
        dt_train_fold_max = dt_train_fold[dt_train_fold["y"] == dt_train_fold["y"].max()]
        dt_train_fold_max = dt_train_fold_max.drop(["y", "Fold"] + cols_cat + cols_remove, axis = 1)
        
        dt_valid_fold_compare = dt_valid_fold.drop(["y", "Fold"] + cols_cat + cols_remove, axis = 1)
        dt_test_compare = dt_all_test.drop(["y", "Fold"] + cols_cat + cols_remove, axis = 1)
        
        # ohe
        distance_valid_ohe = cdist(dt_train_fold_max[cols_encode_ohe], dt_valid_fold_compare[cols_encode_ohe], metric = 'cosine')
        dt_valid_fold["outlierDist_ohe"] = distance_valid_ohe[0]
        distance_test_ohe_temp = cdist(dt_train_fold_max[cols_encode_ohe], dt_test_compare[cols_encode_ohe], metric = 'cosine')
        distance_test_ohe = distance_test_ohe + distance_test_ohe_temp[0]
    
        # label
        distance_valid_label = cdist(dt_train_fold_max[cols_encode_label], dt_valid_fold_compare[cols_encode_label], metric = 'euclidean')
        dt_valid_fold["outlierDist_label"] = distance_valid_label[0]
        distance_test_label_temp = cdist(dt_train_fold_max[cols_encode_label], dt_test_compare[cols_encode_label], metric = 'euclidean')
        distance_test_label = distance_test_label + distance_test_label_temp[0]
        
        # bin
        distance_valid_bin = cdist(dt_train_fold_max[cols_bin], dt_valid_fold_compare[cols_bin], metric = 'cosine')
        dt_valid_fold["outlierDist_bin"] = distance_valid_bin[0]
        distance_test_bin_temp = cdist(dt_train_fold_max[cols_bin], dt_test_compare[cols_bin], metric = 'cosine')
        distance_test_bin = distance_test_bin + distance_test_bin_temp[0]
        
        # tailored bin
        distance_valid_tailoredBin = cdist(dt_train_fold_max[cols_tailoredBin], dt_valid_fold_compare[cols_tailoredBin], metric = 'cosine')
        dt_valid_fold["outlierDist_tailoredBin"] = distance_valid_tailoredBin[0]
        distance_test_tailoredBin_temp = cdist(dt_train_fold_max[cols_tailoredBin], dt_test_compare[cols_tailoredBin], metric = 'cosine')
        distance_test_tailoredBin = distance_test_tailoredBin + distance_test_tailoredBin_temp[0]
        
        # all ohe
        distance_valid_all = distance_valid_ohe + distance_valid_bin
        dt_valid_fold["outlierDist_all"] = distance_valid_all[0]
        distance_test_all_temp = distance_test_ohe_temp[0] + distance_test_bin_temp[0]
        distance_test_all = distance_test_all + distance_test_all_temp[0]
    
        dt_train_outlier = pd.concat([dt_train_outlier, dt_valid_fold])
        
    dt_all_test["outlierDist_ohe"] = distance_test_ohe / max(np.unique(dt_all_train.Fold.values))
    dt_all_test["outlierDist_label"] = distance_test_label / max(np.unique(dt_all_train.Fold.values))
    dt_all_test["outlierDist_bin"] = distance_test_bin / max(np.unique(dt_all_train.Fold.values))
    dt_all_test["outlierDist_tailoredBin"] = distance_test_tailoredBin / max(np.unique(dt_all_train.Fold.values))
    dt_all_test["outlierDist_all"] = distance_test_all / max(np.unique(dt_all_train.Fold.values))
    
    
    return(pd.concat([dt_train_outlier, dt_all_test]))

# clustering
def getClusters(dt_all, cols_cat):
    # cols
#    cols_encode_label = dt_all.filter(regex = "Encode_Label").columns.values.tolist()
    cols_tsne = ['X118',
            'X127',
            'X47',
            'X315',
            'X311',
            'X179',
            'X314',
            'X232',
            'X29',
            'X232',
            'X261']
    
    # standardize
    dt_all_norm = StandardScaler().fit_transform(dt_all[cols_tsne])

    n_comp_tnse = 2
    
    # tsne
    
    tsne = TSNE(random_state=2016,perplexity=50,verbose=2)
    tsne_result = tsne.fit_transform(dt_all_norm)
    dt_tsne = pd.DataFrame({"x1": tsne_result[:, 0], "x2": tsne_result[:, 1]})
    dt_tsne = StandardScaler().fit_transform(dt_tsne)
    
    # mds
    mds = MDS(n_components=n_comp_tnse, random_state = 888)
    mds_result = mds.fit_transform(dt_all_norm)
    
    # Birch
    n_clusters_birch = 2
    birch = Birch(n_clusters = n_clusters_birch)
    birch_result = birch.fit_transform(dt_all_norm)
    
    # kmeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(dt_tsne)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.196, min_samples=100).fit(dt_tsne)
    
    # Append decomposition components to datasets
    for i in range(1, n_comp_tnse + 1):
        dt_all['CL_TSNE_' + str(i)] = tsne_result[:, i - 1]  
        dt_all['CL_MDS_' + str(i)] = mds_result[:, i - 1]
    
    for i in range(1, n_clusters_birch + 1):
        dt_all['CL_BIRCH_' + str(i)] = birch_result[:, i - 1]
    
    for i in np.unique(kmeans.labels_):
        x = kmeans.labels_ == i
        x = x.astype("int64")
        dt_all['CL_Kmeans_' + str(i)] = x
    
    for i in np.unique(dbscan.labels_):
        x = dbscan.labels_ == i
        x = x.astype("int64")
        dt_all['CL_DBSCAN_' + str(i)] = x
        
    
    return(dt_all)