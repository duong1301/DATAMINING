import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score

#Khởi tạo dữ liệu ngẫu nhiên
X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)


#Hiển thị dữ liệu lên đồ thị
plt.scatter(X[:, 0], X[:, 1], c=y, marker="*");
plt.show()

#Tạo tệp train và tệp test theo tỷ lệ 0.67, 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)

# Build a Gaussian Classifier
model = GaussianNB()
# Model training
model.fit(X_train, y_train)
# Predict Output
predicted = model.predict(X_test)
#In kết quả
for i in range(len(y_test)):
    print("Actual Value:", y_test[i], "Predicted Value:", predicted[i])

#Tính và in ra độ chính xác phân lớp, F1 score
accuray = accuracy_score(predicted, y_test)
f1 = f1_score(predicted, y_test, average="weighted")
print("Accuracy:", accuray)
print("F1 Score:", f1)

#In ra confusion matrix
cm = confusion_matrix(y_test, predicted)
print(cm)
