import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_csv('teleCust1000t.csv')
print(df.info())

n = len(df.values[0])
print('Number of columns: ', n)
X = df.values[:, :n-1]
y = df.values[:, n-1:]
y = np.reshape(y, -1)

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

k = 1
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k, metric='hamming').fit(X_train,y_train)
yhat = neigh.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
