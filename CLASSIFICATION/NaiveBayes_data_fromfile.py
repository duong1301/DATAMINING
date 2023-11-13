import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,f1_score,classification_report

#Đọc dữ liệu và in thông tin về dữ liệu
df = pd.read_csv('Loandata.csv')
print(df.info())

#We will now convert the ‘purpose’ column from categorical to integer using pandas `get_dummies` function.
pre_df = pd.get_dummies(df,columns=['purpose'],drop_first=True)
pre_df.head()

X = pre_df.drop('not.fully.paid', axis=1)
y = pre_df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
print(cm)
