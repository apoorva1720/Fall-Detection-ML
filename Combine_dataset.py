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


dataset1 = pd.read_csv('SmartWatch1.csv')
dataset2 = pd.read_csv('SmartWatch2.csv')
dataset3 = pd.read_csv('SmartWatch3.csv')
dataset1['outcome'].values[:] = 0

data = [dataset1, dataset2, dataset3]
datasetA = pd.concat(data)
datasetA.to_csv("SmartWatchCombined.csv")

dataset4 = pd.read_csv('32ms_User1_LeftWrist.csv')
dataset5 = pd.read_csv('32ms_User2_LeftWrist.csv')
dataset6 = pd.read_csv('32ms_User3_LeftWrist.csv')
dataset7 = pd.read_csv('32ms_User4_LeftWrist.csv')
dataset8 = pd.read_csv('32ms_User5_LeftWrist.csv')
dataset9 = pd.read_csv('32ms_User6_LeftWrist.csv')
dataset10 = pd.read_csv('32ms_User7_LeftWrist.csv')

data1 = [dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10]
datasetB = pd.concat(data1)
datasetB.drop(datasetB.columns[4], axis=1, inplace=True)
datasetB.to_csv("NotchCombined.csv")

