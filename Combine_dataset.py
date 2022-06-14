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



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
        
        
knn_file = open(("Comb_Results.csv"), "w")
writer = csv.writer(knn_file)

data_time_s = time.time()
dataset = pd.read_csv('Comb_dataset.csv')
dataset.dropna(axis = 0, inplace = True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
data_time_e = time.time()
data_time = data_time_e-data_time_s
print("Data Loading time: " + str(data_time))


array=[]
for i in range(1,201,2):
	array.append(i)
	training_time_s_1 = time.time()
	classifier_1 = KNeighborsClassifier(n_neighbors = i, metric = "euclidean")
	classifier_1.fit(X_train, y_train)
	scores_1= cross_val_score(classifier_1, X_train, y_train)
	training_time_e_1 = time.time()
	training_time_s_2 = time.time()
	classifier_2 = KNeighborsClassifier(n_neighbors = i, metric = "manhattan")
	classifier_2.fit(X_train, y_train)
	scores_2 = cross_val_score(classifier_2, X_train, y_train)
	training_time_e_2 = time.time()
	training_time_s_5 = time.time()
	classifier_5 = KNeighborsClassifier(n_neighbors = i, metric = "minkowski")
	classifier_5.fit(X_train, y_train)
	scores_5 = cross_val_score(classifier_5, X_train, y_train)
	training_time_e_5 = time.time()
	
	training_time_1 = training_time_e_1-training_time_s_1
	training_time_2 = training_time_e_2-training_time_s_2
	training_time_5 = training_time_e_5-training_time_s_5
	
	array.append(training_time_1)
	array.append(training_time_2)
	array.append(training_time_5)

	testing_time_s_1 = time.time()
	y_pred = classifier_1.predict(X_test)
	accuracy_1 = accuracy_score(y_test,y_pred)
	testing_time_e_1 = time.time()
	
	testing_time_s_2 = time.time()
	y_pred = classifier_2.predict(X_test)
	accuracy_2 = accuracy_score(y_test,y_pred)
	testing_time_e_2 = time.time()
	
	testing_time_s_5 = time.time()
	y_pred = classifier_5.predict(X_test)
	accuracy_5 = accuracy_score(y_test,y_pred)
	testing_time_e_5 = time.time()
	
	testing_time_1 = testing_time_e_1-testing_time_s_1
	testing_time_2 = testing_time_e_2-testing_time_s_2
	testing_time_5 = testing_time_e_5-testing_time_s_5
	
	array.append(testing_time_1)
	array.append(testing_time_2)
	array.append(testing_time_5)
	
	array.append(accuracy_1)
	array.append(accuracy_2)
	array.append(accuracy_5)
	
	writer.writerow(array)
	print(array)
	array = []
	printProgressBar(i, 201, prefix = ' Progress:', suffix = 'Complete', length = 50)
knn_file.close()
