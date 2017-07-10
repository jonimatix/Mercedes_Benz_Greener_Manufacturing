import numpy as np 
import pandas as pd 
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

cm = plt.cm.get_cmap('RdYlBu')


features = ['X118',
            'X127',
            'X47',
            'X315',
            'X311',
            'X179',
            'X314',
### added by Tilii
            'X232',
            'X29',
            'X263',
###
            'X261']


train = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/train.csv")
test = pd.read_csv("/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/raw/test.csv")

y_clip = np.clip(train['y'].values, a_min=None, a_max=130)


tsne = TSNE(random_state=2016,perplexity=50,verbose=2)
x = tsne.fit_transform(pd.concat([train[features],test[features]]))


plt.figure(figsize=(12,10))
# plt.scatter(x[train.shape[0]:,0],x[train.shape[0]:,1], cmap=cm, marker='.', s=15, label='test')
cb = plt.scatter(x[:train.shape[0],0],x[:train.shape[0],1], c=y_clip, cmap=cm, marker='o', s=15, label='train')
plt.colorbar(cb)
plt.legend(prop={'size':15})
#plt.title('t-SNE embedding of train & test data', fontsize=20)
plt.title('t-SNE embedding of train data', fontsize=20)

X = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1]})
dt_all_norm = StandardScaler().fit_transform(X)

# dbscan
dbscan = DBSCAN(eps=0.196, min_samples=100).fit(dt_all_norm)
print(np.unique(dbscan.labels_))

plt.scatter(dt_all_norm[:4029, 0], dt_all_norm[:4029, 1], c=dbscan.labels_[:4029]
, cmap=cm, marker='o', s=15, label='train')
plt.show()


# kmeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(dt_all_norm)
plt.scatter(dt_all_norm[:4029, 0], dt_all_norm[:4029, 1], c=kmeans.labels_[:4029]
, cmap=cm, marker='o', s=15, label='train')
plt.show()

for i in np.unique(kmeans.labels_):
    
x = kmeans.labels_ == 0
x.astype("int64")