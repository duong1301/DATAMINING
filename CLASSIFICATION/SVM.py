from sklearn import datasets #để lấy bộ data
from sklearn.model_selection import train_test_split #Để chia bộ data thành tệp train và test
from sklearn import svm # để làm mô hình
from sklearn import metrics #để đo kết quả phân lớp

#Tải dataset, sử dụng bộ breastcancer
cancer = datasets.load_breast_cancer()
#In thông tin của bộ dữ liệu

#Tạo bộ train và bộ test
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)

#Tạo mô hình
clf = svm.SVC(kernel='linear') #Linear Kernel
#Huấn luyện
clf.fit(X_train, y_train)
#Sử dụng mô hình để phân lớp
y_pred = clf.predict(X_test)
#Kiểm tra kết quả
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))