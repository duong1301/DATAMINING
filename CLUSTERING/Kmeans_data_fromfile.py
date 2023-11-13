from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

dataset = np.loadtxt("iris.csv", delimiter=",", skiprows=1)
n = len(dataset[0])

x = dataset[:,:n-1]
y = dataset[:, n-1:]
y = np.reshape(y, -1)

print('Data: ', x)
print('The original group label: ', y)




kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

for i in range(len(y)):
    print(y[i],' - ', kmeans.labels_[i])

print("Accuracy:",metrics.accuracy_score(y, kmeans.labels_))

ari = adjusted_rand_score(y, kmeans.labels_)
print('Rand index: ', ari)


y = y.reshape(150, 1)
ss = silhouette_score(y, kmeans.labels_)
print('Silhouette score: ', ss)