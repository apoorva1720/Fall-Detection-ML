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
algorithms = ["KNN", "Logistic Regression", "Naive Bayes", "Random Forest"]

#Defining data loading function for single thread execution
def _LoadData_SingleThread(array):
	data_time_s = time.time()
	dataset = pd.read_csv('SmartFall.csv')
	dataset.drop(dataset.columns[0], axis=1, inplace=True)
	
	dataset.dropna(axis = 0, inplace = True)
	X_train = dataset.iloc[:, :-1].values
	y_train = dataset.iloc[:,-1].values
	
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
	
	dataset1 = pd.read_csv('SmartWatch1.csv')
	dataset2 = pd.read_csv('SmartWatch2.csv')
	dataset3 = pd.read_csv('SmartWatch3.csv')
	dataset1['outcome'].values[:] = 0

	data = [dataset1, dataset2, dataset3]
	dataset4 = pd.concat(data)

	dataset4.dropna(axis = 0, inplace = True)
	X = dataset4.iloc[:, :-1].values
	y = dataset4.iloc[:,-1].values
	
	dataset5 = pd.read_csv('32ms_User1_LeftWrist.csv')
	dataset6 = pd.read_csv('32ms_User2_LeftWrist.csv')
	dataset7 = pd.read_csv('32ms_User3_LeftWrist.csv')
	dataset8 = pd.read_csv('32ms_User4_LeftWrist.csv')
	dataset9 = pd.read_csv('32ms_User5_LeftWrist.csv')
	dataset10 = pd.read_csv('32ms_User6_LeftWrist.csv')
	dataset11 = pd.read_csv('32ms_User7_LeftWrist.csv')

	data1 = [dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11]
	dataset12 = pd.concat(data1)

	dataset12.dropna(axis = 0, inplace = True)
	X_1 = dataset12.iloc[:, :-2].values
	y_1 = dataset12.iloc[:,-2].values

	X_test = [X, X_1]
	y_test = [y, y_1]
	
	X_test = pd.concat(X_test)
	y_test = pd.concat(y_test)
	
	
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
		print("ok")
		classifier = SVC(C=10, kernel = 'rbf', random_state = 0, probability=True)
		print('ok')
		classifier.fit(X_train, y_train)
		print('ok')
		scores = cross_val_score(classifier, X_train, y_train, cv=10)
		
		
	elif ModelName == "KNN":
		classifier = KNeighborsClassifier(n_neighbors = 30)
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
	array.append(accuracy)
	array.append(sensitivity)
	array.append(false_rate)
	array.append(specificity)
	

for algo in algorithms:
	f = open(("Results_" + algo + "final.csv"), "w")
	writer = csv.writer(f)
	print(algo)
	for u in tqdm(range(epochs)):
		# loadData
		array =[]
		X_train, X_test, y_train, y_test = _LoadData_SingleThread(array)
		# trainModel
		classifier = _TrainModel_SingleThread(X_train, X_test, y_train, y_test, algo, array)
		#testModel
		_TestModel_SingleThread(classifier, array)
		writer.writerow(array)
	f.close()

