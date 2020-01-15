import sys
import pickle
import logging
import datetime

from os import path
import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils import save
from utils import load
from utils import loadJSON
from utils import loadCategorialSeries
from utils import labelCategoricalColumn

CHUNKSIZE = 1000000
CONSTANT_FILLER = 'unknown_flag'


def predict(parameter_file):

	XGBModel = load(parameter_file.model_filepath)
	if XGBModel is None:
		raise RuntimeError("Filename of model "+parameter_file.model_filepath+" is incorrect")

	# Load the columns dictionary
	columns_to_check = { 'YColumns': [], 'labelCols': [], 'categoricalCols': [], 'numericCols': []}
	if parameter_file.columns_filepath is not None:
		columns_to_check = load(parameter_file.columns_filepath)

	for col in columns_to_check.categoricalCols:
		tempSeries, tempList = loadCategorialSeries(base_folder, col)
		categoryLists.append(tempList)

	one_hot_encoder = OneHotEncoder(categories=categoryLists, handle_unknown='ignore', sparse=False)	

	labelSeries = {}
	for col in columns_to_check.labelCols:
		labelSeries[''+col], tempList = loadCategorialSeries(base_folder, col)

	min_max_dict = {}
	for chunk in pd.read_csv(parameter_file.testing_data_file, chunksize=CHUNKSIZE):

		columns_to_keep = columns_to_check.YColumns + columns_to_check.categoricalCols + columns_to_check.labelCols 
				+ columns_to_check.numericCols

		df_merged_set = chunk[columns_to_keep]	
		
		for col in labelCols:
			df_merged_set[col] = labelCategoricalColumn(df_merged_set, col, labelSeries[col])
			df_merged_set[col] = (df_merged_set[col]/df_merged_set[col].max())

		nLength = len(numericCols)
		cLength = len(categoricalCols)
		nLabellength = len(labelCols)
		X1, X2 = df_merged_set.iloc[:,1:cLength+1], df_merged_set.iloc[:,cLength+1:cLength+nLabellength+nLength+1], 
		Y = df_merged_set.iloc[:,0]
		
		one_hot_encoder.fit(X1)
		one_hot_encoded = one_hot_encoder.transform(X1)

		dataMatrix = xgb.DMatrix(np.concatenate((X2, one_hot_encoded), axis=1), label=Y.to_numpy())
		predictions = xg_reg.predict(dataMatrix)

		df = pd.DataFrame({'actual': Y, 'predictions': predictions})
		for	row in df.iterrows():
			if (min_max_dict[''+row['actual']] is None):
				min_max_dict({'min': row['predictions'], 'max': row['predictions']})
			elif (row['predictions']>min_max_dict[''+row['actual']].max):
				min_max_dict[''+row['actual']].max = row['predictions']
			elif (row['predictions']<min_max_dict[''+row['actual']].min):
				min_max_dict[''+row['actual']].min = row['predictions']

		accuracy = r2_score(Y, predictions)

	save('output_{}.sav', parameter_file.testing_data_file,  min_max_dict)		

def __main__():
	# count the arguments
	if len(sys.argv) < 2:
		raise RuntimeError("Please provide the parameter_file")

	params_json = loadJSON(sys.argv[1])
	predict(params_json)

#This is required to call the main function
if __name__ == "__main__":
	__main__()






