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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# This will only use the Label Encoder as on using the one hot encoding 
# we have to guarantee all the values in that one-hot-column has to present 
# in the test dataset also otherwise it will result in the feature mismatch bug

# TODO To resolve this we need put the columns in same order and intialize the old columns 
# We will do this in later in this we are training a simple XGB Model for the seven days data

# Batch size of 100000
CHUNKSIZE = 10000
CONSTANT_FILLER = 'unknown_flag'

def saveModel(xg_reg, learning_rate_val, max_depth_val):
	filename =  '/data/models/XGB_MODEL_{}_{}_{}.sav'
	filename  = filename.format(learning_rate_val, max_depth_val, int(datetime.datetime.now().timestamp())) 
	pickle.dump(xg_reg, open(filename, 'wb'))

	logging.info("training complete and model is saved")

# Load the Labels Vocabulary in One Hot Encoder
def loadCategorialList(base_folder, columnNm):
	map_file = base_folder + columnNm + "_map.dict"
	if not path.exists(map_file):
		raise RuntimeError("Map file missing for "+ columnNm + " name")

	column_dict_file = open(map_file, "rb")
	column_dict = pickle.load(column_dict_file)
	column_series = pd.Series(column_dict)
	return list(column_series.index)

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
	categoricalCols = [ 'slot_names', 'container_type', 'component_name', 'component_namespace', 'site']
	categoryLists = []
	for col in categoricalCols:
		categoryLists.append(loadCategorialList(base_folder, col))

	one_hot_encoder = OneHotEncoder(categories=categoryLists, handle_unknown='ignore', sparse=False)	

	chunkcount = 1
	training_data_file = base_folder + '3HourDataFullFile.csv'
	for chunk in pd.read_csv(training_data_file, chunksize=CHUNKSIZE):
		logging.info("Start chunk Processing - " + str(chunkcount))
		logging.info(chunk.columns)
		
		YColumns = ['impressions']
		numericCols = ['guarantee_percentage', 'days_interval', 'hours_interval']
		columns_to_keep = YColumns + categoricalCols + numericCols 
		df_merged_set = chunk[columns_to_keep]	
		
		nLength = len(numericCols)
		cLength = len(categoricalCols)
		X1, X2, Y = df_merged_set.iloc[:,1:cLength+1], df_merged_set.iloc[:,cLength+1:cLength+nLength+1], df_merged_set.iloc[:,0]
		
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
	saveModel(xg_reg, learning_rate_val, max_depth_val)
	predict(xg_reg, one_hot_encoder, base_folder, earlyBreak)

def predict(xg_reg, one_hot_encoder, base_folder, earlyBreak):
	training_data_file = base_folder + '3HourDataFullFile.csv'

	chunkcount = 1
	categoricalCols = [ 'slot_names', 'container_type', 'component_name', 'component_namespace', 'site']
	for chunk in pd.read_csv(training_data_file, chunksize=CHUNKSIZE):
		YColumns = ['impressions']
		numericCols = ['guarantee_percentage', 'days_interval', 'hours_interval']
		categoricalCols = [ 'slot_names', 'container_type', 'component_name', 'component_namespace', 'site']

		columns_to_keep = YColumns + categoricalCols + numericCols 
		df_merged_set = chunk[columns_to_keep]	
		
		nLength = len(numericCols)
		cLength = len(categoricalCols)
		X1, X2, Y = df_merged_set.iloc[:,1:cLength+1], df_merged_set.iloc[:,cLength+1:cLength+nLength+1], df_merged_set.iloc[:,0]
		
		one_hot_encoder.fit(X1)
		one_hot_encoded = one_hot_encoder.transform(X1)

		dataMatrix = xgb.DMatrix(np.concatenate((X2, one_hot_encoded), axis=1), label=Y.to_numpy())
		predictions = xg_reg.predict(dataMatrix)

		df = pd.DataFrame({'actual': Y, 'predictions': predictions})
		accuracy = r2_score(Y.to_numpy(), predictions)
		logging.info("Accuracy(Max:1 , 0 for stright line) : " + str(accuracy))
		logging.info(str(df.head()))
		
		if(earlyBreak=='1' and chunkcount>10):
			break
		chunkcount = chunkcount + 1 	
	logging.info("Prediction Over")

def __main__():
	# count the arguments
	if len(sys.argv) < 4:
		raise RuntimeError("Please provode the learning_rate, max_depth, base folder and earlyBreak")
	trainModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

#This is required to call the main function
if __name__ == "__main__":
	__main__()