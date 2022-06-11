import pandas as pd
import numpy as np
import os
import glob
import math
import csv
import time
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
        
        
def analyse_file(filename):
	#Reading the data file
	data = pd.read_csv(filename , names=range(8))
	#Filtering Sensor Data
	accelerometer_data = data[data[5] == "Cywee Accelerometer Sensor"]
	accelerometer_data=accelerometer_data.drop([1,5,6,7], axis = 1)
	accelerometer_data=accelerometer_data.rename(columns={0: "TimeStamp", 2: "AX", 3: "AY", 4: "AZ"})
	accelerometer_data = accelerometer_data.astype(float)
	ax = accelerometer_data["AX"] * accelerometer_data["AX"]
	ay = accelerometer_data["AY"] * accelerometer_data["AY"]
	az = accelerometer_data["AZ"] * accelerometer_data["AZ"]
	am = ax + ay + az
	am = am.apply(lambda x: math.sqrt(x))
	accelerometer_data["ARMS"] = am

	#Statistical analysis for all Data
	parameters = Statistical_Analysis(accelerometer_data)
	#return all parameters calculated for file
	return parameters
	
	
def Statistical_Analysis(accelerometer_data):

	parameters = []
	#Calculating Parameters for Accelerometer
	for column in ["AX", "AY", "AZ", "ARMS"]:
		parameters.extend(stat_analysis_column(accelerometer_data[column]))
	return parameters

def stat_analysis_column(data):
	features = [
	data.mean(skipna = True),
	data.std(skipna = True),
	data.var(skipna = True),
	data.min(skipna = True),
	data.max(skipna = True),
	data.skew(skipna = True),
	data.kurtosis(skipna = True)]
	#Add code for spectral entropy
	return features



data_source_path = "/Users/apoorva/Desktop/INSTRSOP/Fall-Detection-ML/TheOne/*"
data_destination_path = "/Users/apoorva/Desktop/INSTRSOP/Fall-Detection-ML/Acc_Features.csv"
i=0
l=389
nan_list = []
start=time.time()
Fall_occurance = 0
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
with open(data_destination_path, 'w') as out_file:
	rows = csv.writer(out_file)
	
	list_of_folders = glob.glob(data_source_path)
	for folder in list_of_folders:
		if "FallData" in folder:
			Fall_occurance = 1
		else:
			Fall_occurance = 0
		list_of_folders_1 = glob.glob(folder + "/*")
		for sub_folder in list_of_folders_1:
			list_of_files = glob.glob(sub_folder + "/*.csv")
			for file in list_of_files:
				parameters = analyse_file(file)
				if True in pd.isna(parameters):
					nan_list.append(file)
				rows.writerow(parameters + [Fall_occurance])
				#print("Extraction Completed: " + file)
				i += 1
				printProgressBar(i, l, prefix = ' Progress:', suffix = 'Complete', length = 50)
end=time.time()
print(end-start)

