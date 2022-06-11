import threading
import pandas as pd
import numpy
import time
import csv
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
#algo = "SVM"mo
algorithms = ["SVM", "KNN", "Logistic Regression", "Naive Bayes", "Random Forest"]

#Defining data loading function for single thread execution
def _LoadData_SingleThread(array):
	data_time_s = time.time()
	dataset = pd.read_csv('Features.csv')
	dataset.dropna(axis = 0, inplace = True)
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:,-1].values
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	data_time_e = time.time()
	data_time = data_time_e-data_time_s
	print("Data Loading time: " + str(data_time))
	array.append(data_time)
	return X_train, X_test, y_train, y_test

#Defining ML training function for single thread execution
def _TrainModel_SingleThread(X_train, X_test, y_train, y_test, ModelName, array):
	training_time_s = time.time()
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
	training_time_e = time.time()
	training_time = training_time_e-training_time_s
	print("Training time: " + str(training_time))
	array.append(training_time)
	return classifier




#Defining ML testing function for single thread execution
def _TestModel_SingleThread(classifier, array):
	testing_time_s = time.time()
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
	testing_time_e = time.time()
	testing_time = testing_time_e-testing_time_s
	print("Testing time: " + str(testing_time))
	print("Acc " + str(accuracy))
	print("Sens "+ str(sensitivity))
	print("fals " + str(false_rate))
	print("Specs " + str(specificity))
	array.append(testing_time)

	

for algo in algorithms:
	#f = open(("Results_" + algo + "_i5.csv"), "w")
	#writer = csv.writer(f)
	print(algo)
	for u in tqdm(range(epochs)):
		# loadData
		array =[]
		X_train, X_test, y_train, y_test = _LoadData_SingleThread(array)
		# trainModel
		classifier = _TrainModel_SingleThread(X_train, X_test, y_train, y_test, algo, array)
		#testModel
		_TestModel_SingleThread(classifier, array)
		#writer.writerow(array)
	#f.close()




