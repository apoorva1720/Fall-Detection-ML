import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

dataset = pd.read_csv('Features.csv')
dataset.dropna(axis = 0, inplace = True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
start = time.time()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C=10, kernel = 'rbf', random_state = 0, probability=True)
classifier.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=10)
print(scores,scores.mean())
end=time.time()
import joblib
# Save the model as a pickle in a file
joblib.dump(classifier, 'SVM.pkl')
# Load the pickled model
start1=time.time()
Test_Classifier = joblib.load('SVM.pkl')

# Predicting the Test set results
y_pred = Test_Classifier.predict(X_test)
end1=time.time()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
print(classification_report(y_test, y_pred))
print("The accuracy of the model is  {} %".format(str(round(accuracy_score(y_test,y_pred),4)*100)))
print("Training time=")
print(end-start)
print("Testing time=")
print(end1-start1)


y_pred_proba = Test_Classifier.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
