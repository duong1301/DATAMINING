import pandas as pd #để đọc dữ liệu
from sklearn.model_selection import train_test_split#để tạo bộ train và test
from sklearn.svm import SVC #để tạo mô hình SVM
from sklearn.metrics import classification_report, confusion_matrix#để báo cáo kết quả
#Đọc dữ liệu từ file, chú ý đường
bankdata = pd.read_csv("IRIS.csv")
print(bankdata)
print("Kích thước: ", bankdata.shape)
print("Dữ liệu:\n",   bankdata.head())
x = bankdata.drop('CLASS', axis=1)
y = bankdata['CLASS']
print('x: ', x)
print('Y: ', y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=True)
x_train = x
y_train = y
x_test = x_train
y_test = y_train

svclassifier = SVC(kernel='rbf', C=10, gamma=0.01)
svclassifier.fit(x_train, y_train)
#Sử dụng mô hình để phân lớp
y_pred = svclassifier.predict(x_test)
#Kiểm tra kết quả
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))