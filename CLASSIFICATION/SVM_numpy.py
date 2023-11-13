from sklearn import datasets #để lấy bộ data
from sklearn.model_selection import train_test_split #Để chia bộ data thành tệp train và test
from sklearn import svm # để làm mô hình
from sklearn import metrics #để đo kết quả phân lớp
import numpy as np

#Đọc tệp dữ liệu
dataset = np.loadtxt("iris.csv", delimiter=",", skiprows=1)
n = len(dataset[0])
print(n)
X_train = dataset[:,:n-1]
y_train = dataset[:, n-1:]

y_train = y_train.reshape(len(y_train))
print(X_train)
print(y_train)

X_test, y_test = X_train, y_train
#Tạo mô hình
clf = svm.SVC(kernel='rbf') # Linear Kernel
#Huấn luyện
clf.fit(X_train, y_train)
#Sử dụng mô hình để phân lớp
y_pred = clf.predict(X_test)
#Kiểm tra kết quả
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='micro'))
print("Recall:",metrics.recall_score(y_test, y_pred, average='micro'))