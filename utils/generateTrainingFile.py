import sys
import sys
import pickle
import logging
import datetime

from os import path
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Import all utilitty function
from utils import *

# Batch size of 10000000
CHUNKSIZE = 500000
CONSTANT_FILLER = 'unknown_flag'


def trainModel(learning_rate_val, max_depth_val, base_folder, earlyBreak):		
	learning_params = {
		'objective' : 'reg:squarederror',
		'colsample_bytree' : 0.3,
		'learning_rate' : learning_rate_val, 
        'max_depth' : max_depth_val,
        'alpha' : 5,
        'n_estimators' : 10
	}
	xg_reg = None

	#Load the categorical columns for faster filling in between
	categoricalCols = [ 'container_type', 'site', 'language_code']
	categoryLists = []
	for col in categoricalCols:
		tempSeries, tempList = loadCategorialSeries(base_folder, col)
		categoryLists.append(tempList)

	one_hot_encoder = OneHotEncoder(categories=categoryLists, handle_unknown='ignore', sparse=False)	

	labelCols = ['slot_names', 'component_name', 'component_namespace']
	labelSeries = {}
	for col in labelCols:
		labelSeries[''+col], tempList = loadCategorialSeries(base_folder, col)

	chunkcount = 1
	training_data_file = base_folder + 'full_7_day_ML.csv'
	for chunk in pd.read_csv(training_data_file, chunksize=CHUNKSIZE):
		logging.info("Start chunk Processing - " + str(chunkcount))
		logging.info(chunk.columns)
		
		YColumns = ['impressions']
		numericCols = ['guarantee_percentage', 'start_days', 'start_hours']
		columns_to_keep = YColumns + categoricalCols + labelCols + numericCols 
		df_merged_set = chunk[columns_to_keep]	
		
		for col in labelCols:
			df_merged_set[col] = labelCategoricalColumn(df_merged_set, col, labelSeries[col])

		nLength = len(numericCols)
		cLength = len(categoricalCols)
		nLabellength = len(labelCols)
		X1, X2 = df_merged_set.iloc[:,1:cLength+1], df_merged_set.iloc[:,cLength+1:cLength+nLabellength+nLength+1], 
		Y = df_merged_set.iloc[:,0]
		# Use label encoding for bigger columns

		one_hot_encoder.fit(X1)
		one_hot_encoded = one_hot_encoder.transform(X1)

		# Drop the ctegorical columns 	
		dataMatrix = xgb.DMatrix(np.concatenate((X2, one_hot_encoded), axis=1), label=Y.to_numpy())
		if(chunkcount==1):
			xg_reg = xgb.train(learning_params, dataMatrix, 10)
		else:
			# Takes in the intially model and produces a better one
			xg_reg = xgb.train(learning_params, dataMatrix, 10, xgb_model=xg_reg)
		logging.info("Model saved "+str(xg_reg))
		if(earlyBreak=='1' and chunkcount>10):
			break
		chunkcount = chunkcount + 1 
	
	logging.info(xg_reg)
	saveModel(xg_reg, learning_rate_val, max_depth_val, base_folder)
	predict(xg_reg, one_hot_encoder, base_folder, earlyBreak, labelSeries)

def __main__():
	# count the arguments
	if len(sys.argv) < 5:
		raise RuntimeError("Please provode the learning_rate, max_depth and base folder")
	trainModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

#This is required to call the main function
if __name__ == "__main__":
	__main__()