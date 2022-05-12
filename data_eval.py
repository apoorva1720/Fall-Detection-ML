import pandas as pd
import numpy as np
import os
import glob
import math
import csv
import matplotlib.pyplot as plt

SVM = {}
KNN = {}
LR = {}
NB = {}
RF = {}

def store_data(dict, algorithm):
	list_of_files = glob.glob(algorithm + "/*.csv")
	for file in list_of_files:
		data = pd.read_csv(file)
		if "M1" in file:
			dict["M1"] = data
		if "i5" in file:
			dict["i5"] = data
		if "i7" in file:
			dict["i7"] = data
		if "410c" in file:
			dict["410c"] = data



data_source_path = ""
list_of_folders = glob.glob(data_source_path)
for algorithm in list_of_folders:
	if "SVM" in algorithm:
		store_data(SVM, algorithm)
	elif "KNN" in algorithm:
		store_data(KNN, algorithm)
	elif "LR" in algorithm:
		store_data(LR, algorithm)
	elif "NB" in algorithm:
		store_data(NB, algorithm)
	elif "RF" in algorithm:
		store_data(RF, algorithm)
			
