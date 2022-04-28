import threading
import pandas as pd
import numpy
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

sensitivity = 0
false_rate = 0
specificity = 0
#Defining global variables
epochs = 1
#algorithms = ["SVM"]
algorithms = ["SVM", "KNN", "Logistic Regression", "Naive Bayes", "Random Forest"]

#Defining data loading function for single thread execution
def _LoadData_SingleThread():

	dataset = pd.read_csv('Features.csv')
	dataset.dropna(axis = 0, inplace = True)
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:,-1].values
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	return X_train, X_test, y_train, y_test

#Defining ML training function for single thread execution
def _TrainModel_SingleThread(X_train, X_test, y_train, y_test, ModelName):

	if ModelName == "SVM":
		classifier = SVC(C=10, kernel = 'rbf', random_state = 0, probability=True)
		classifier.fit(X_train, y_train)
		scores = cross_val_score(classifier, X_train, y_train, cv=10)
		
	elif ModelName == "KNN":
		classifier = KNeighborsClassifier(n_neighbors = 10)
		classifier.fit(X_train, y_train)
		scores = cross_val_score(classifier, X_train, y_train, cv=10)
	
	elif ModelName == "Logistic Regression":
		classifier = LogisticRegression()
		classifier.fit(X_train, y_train)
		scores = cross_val_score(classifier, X_train, y_train, cv=10)
	
	elif ModelName == "Naive Bayes":
		classifier = GaussianNB()
		classifier.fit(X_train, y_train)
		scores = cross_val_score(classifier, X_train, y_train, cv=10)
	
	elif ModelName == "Random Forest":
		classifier = RandomForestClassifier(n_estimators = 100)
		classifier.fit(X_train, y_train)
		scores = cross_val_score(classifier, X_train, y_train, cv=10)

	return classifier




#Defining ML testing function for single thread execution
def _TestModel_SingleThread(classifier):
	y_pred = classifier.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	tp = cm[0][0]
	fp = cm[0][1]
	fn = cm[1][0]
	tn = cm[1][1]
	sensitivity = tp / (tp+tn)
	false_rate = fn / (tp+fn)
	specificity = tn /(tn+fp)
	accuracy = accuracy_score(y_test,y_pred)
	print(sensitivity)
	print(false_rate)
	print(specificity)
	print(cm)
	print(accuracy)


	



with tqdm(range(epochs)) as epochLoop:
	for _ in epochLoop:
		for algo in algorithms:
			print("######################################")
			print("Starting Algorithm test: " + algo)
			# loadData
			X_train, X_test, y_train, y_test = _LoadData_SingleThread()
    
			# trainModel
			classifier = _TrainModel_SingleThread(X_train, X_test, y_train, y_test, algo)
    
			#testModel
			_TestModel_SingleThread(classifier)
			print("Finished Algorithm test: " + algo)
			print("######################################")





