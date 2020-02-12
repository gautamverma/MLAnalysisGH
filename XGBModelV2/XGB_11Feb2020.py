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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
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

# Batch size of 10000
CHUNKSIZE = 1000000
TRAIN_ITERATION = 30

CONSTANT_FILLER = 'missing'
ALL_CONSUMER    = 'allCustomer'
IMPRESSION_COUNT = 10

def mergeDataframe(df1, df2, column, joinType='inner'):
	if column is None:
		raise RuntimeError("Column can't be null. Please give the column value")
	return pd.merge(df1, df2, on=column, how=joinType);

def loadAndMerge(files):
	# placement_metrics_file  {0}
	# placement_metadata_file {1}
	# content_metadata_file {2}
	# placement_properties_file {3}
	# creative_metadata_file {4}

	df1 = pd.read_csv(files[0], skiprows=0, header=0)
	df2 = pd.read_csv(files[1], skiprows=0, header=0)
	df3 = pd.read_csv(files[2], skiprows=0, header=0)
	df4 = pd.read_csv(files[3], skiprows=0, header=0)
	df5 = pd.read_csv(files[4], skiprows=0, header=0)

	logging.info('File Loaded');

	df_merged_set = pd.merge(df1, df4, on='frozen_placement_id', how='inner')
	df_merged_set = pd.merge(df_merged_set, df2, on='frozen_placement_id', how='inner')
	df_merged_set = pd.merge(df_merged_set, df3, on='frozen_content_id', how='inner')
	df_merged_set = pd.merge(df_merged_set, df5, on='creative_id', how='inner')
	logging.info('File merged');
	return df_merged_set


def label_column(df, column):
	# TODO Make it general
	unique_container_id_list = df.container_id.unique().tolist()
	
	unique_container_id_hash = {}
	position = 1
	for val in unique_container_id_list:
		unique_container_id_hash[val] = position
		position = position + 1

	df['container_id_label'] = df.apply (lambda row: unique_container_id_hash[row[column]], axis=1)
	logging.info('Label done for '+ column)
	return df

def generateCleanFile(files, training_file_name):

	if path.exists(training_file_name):
		logging.info("Training file is already present")
		return

	df_merged_set = loadAndMerge(files)
	logging.info("Loading and dataset merged. Display head-- ")
	logging.info(df_merged_set.head())

	# Clean few Columns 
	# Targetting Columns
	df_merged_set[['customer_targeting']] = df_merged_set[['customer_targeting']].fillna(value=ALL_CONSUMER)
	df_merged_set[['guarantee_percentage']] = df_merged_set[['guarantee_percentage']].fillna(value=CONSTANT_FILLER)
	df_merged_set[['component_display_name']] = df_merged_set[['component_display_name']].fillna(value=CONSTANT_FILLER)
	logging.info('Targetting Columns Cleaned');
	# Creative Columns
	df_merged_set[['objective']] = df_merged_set[['objective']].fillna(value=CONSTANT_FILLER)
	df_merged_set[['intent']] = df_merged_set[['intent']].fillna(value=CONSTANT_FILLER)
	logging.info('Creative Columns Cleaned');

	# Generate the unique set and map values
	df_merged_set = label_column(df_merged_set, 'container_id')

	logging.info("Dataframe Shape "+str(df_merged_set.shape))
	df_merged_set.to_csv(training_file_name, index=False, encoding='utf-8')
	logging.info('File Created')


def label_result(row):
	if(row['impressions']>IMPRESSION_COUNT):
		return 1
	return 0

def removeNaN(df, categoricalCols):
	# Replace any NaN values
	for col in categoricalCols:
		df[[col]] = df[[col]].fillna(value=CONSTANT_FILLER)
	return df

# Build the One hot encoder using all data
def buildOneHotEncoder(training_file_name, categoricalCols):
	one_hot_encoder = OneHotEncoder(sparse=False)

	df = pd.read_csv(training_file_name, skiprows=0, header=0)
	df = df[categoricalCols]
	df = removeNaN(df, categoricalCols)
	logging.info(df.shape)
	one_hot_encoder.fit(df)
	return one_hot_encoder

def trainModel(learning_rate, max_depth, training_file_name):

	learning_params = {
	    'objective' : 'binary:logistic',
	    'colsample_bytree' : 0.3,
	    'learning_rate' : learning_rate, 
	    'max_depth' : max_depth,
	    'alpha' : 5,
	    'n_estimators' : 200
	}

	# Init a base Model
	xg_reg = {}

	YColumns = ['result']
	numericalCols = ['impressions', 'guarantee_percentage', 'container_id_label']
	categoricalCols = [ 'slot_names', 'container_type', 'component_name', 'component_namespace',
						'component_display_name', 'customer_targeting', 'site']

	startOneHotIndex = len(numericalCols)
	columns_to_keep = YColumns + numericalCols + categoricalCols

	OneHotEncoder = buildOneHotEncoder(training_file_name, categoricalCols)
	logging.info('One hot encoder')

	chunkcount = 1
	for chunk in pd.read_csv(training_file_name, chunksize=CHUNKSIZE):
		# Train on a part of dataset and predict on other
		if(chunkcount>TRAIN_ITERATION):
			break
		logging.info('Starting Training - '+str(chunkcount))
		chunk['result'] = chunk.apply (lambda row: label_result(row), axis=1)

		# Get all rows where weblab is missing
		df_merged_without_weblab = chunk.where(chunk['weblab']=="missing")
		df_merged_set_test = df_merged_without_weblab[columns_to_keep]
		logging.info('Weblab Removed')

		logging.info("Shape before removal " + df_merged_set_test.shape);
		df_merged_set_test = removeNaN(df_merged_set_test, categoricalCols)
		logging.info(df_merged_set_test.shape);
		INPUT, OUTPUT = df_merged_set_test.iloc[:,1:], df_merged_set_test.iloc[:,0]

		logging.info(str(INPUT.head()))
		logging.info(str(startOneHotIndex))
		one_hot_encoded = OneHotEncoder.transform(INPUT.iloc[:,startOneHotIndex:])
		logging.info('One hot encoding done')
		dataMatrix = xgb.DMatrix(np.column_stack((INPUT.iloc[:,1:startOneHotIndex], one_hot_encoded)), label=OUTPUT)

		if(chunkcount==1):
			xg_reg = xgb.train(learning_params, dataMatrix, 200)
		else:
			# Takes in the intially model and produces a better one
			xg_reg = xgb.train(learning_params, dataMatrix, 200, xgb_model=xg_reg)
		chunkcount = chunkcount + 1
		logging.info("Model saved "+str(xg_reg))

	saveModel(xg_reg, learning_rate, max_depth, columns_to_keep)
	predict(training_file_name, OneHotEncoder, xg_reg)
	return

def saveModel(xg_reg, learning_rate_val, max_depth_val, columns_to_keep):

	model_filename =  '/data/models/XGB_MODEL_{}_{}_{}.sav'
	timestamp_value = int(datetime.datetime.now().timestamp())
	model_filename  = model_filename.format(learning_rate_val, max_depth_val, timestamp_value) 
	pickle.dump(xg_reg, open(model_filename, 'wb'))
	
	column_filename =  '/data/models/XGB_MODEL_COLUMN_{}.sav'
	column_filename = column_filename.format(timestamp_value)
	pickle.dump(columns_to_keep, open(column_filename, 'wb'))
	logging.info("Model and columns are saved")

def predict(training_file_name, OneHotEncoder, xg_reg):

	YColumns = ['result']
	numericalCols = ['impressions', 'guarantee_percentage', 'container_id_label']
	categoricalCols = [ 'slot_names', 'container_type', 'component_name', 'component_namespace',
						'component_display_name', 'customer_targeting', 'site']

	startOneHotIndex = len(numericalCols)
	columns_to_keep = YColumns + numericalCols + categoricalCols

	chunkcount = 1
	for chunk in pd.read_csv(training_file_name, chunksize=CHUNKSIZE):
		if(chunkcount<=TRAIN_ITERATION):
			chunkcount = chunkcount + 1
			continue

		chunk['result'] = chunk.apply (lambda row: label_result(row), axis=1)
		# Get all rows where weblab is missing
		df_merged_without_weblab = chunk.where(chunk['weblab']=="missing")
		df_merged_set_test = df_merged_without_weblab[columns_to_keep]
		
		df_merged_set_test = removeNaN(df_merged_set_test, categoricalCols)
		INPUT, OUTPUT = df_merged_set_test.iloc[:,1:], df_merged_set_test.iloc[:,0]
		
		one_hot_encoded = OneHotEncoder.transform(INPUT.iloc[:,startOneHotIndex:])
		
		dataMatrix = xgb.DMatrix(np.column_stack((INPUT.iloc[:,1:startOneHotIndex], one_hot_encoded)), label=OUTPUT)

		predictions = xg_reg.predict(dataMatrix)

		# Result Analysis for Chunk
		results = pd.DataFrame({'actual': OUTPUT, 'predict': predictions})
		matrix = confusion_matrix(OUTPUT, np.around(predictions))
		print('Accuracy Score :',accuracy_score(OUTPUT, np.around(predictions))) 
		print('Report : ')
		print(classification_report(OUTPUT, np.around(predictions))) 

	return

def startSteps(learning_rate, max_depth):
	files =	[ 
			'/data/s3_file/FE/18January03FebPMetrics000',
			'/data/s3_file/FE/18January03FebPMetadata000',
			'/data/s3_file/FE/18January03FebCM000',
			'/data/s3_file/FE/18January03FebPP000',
			'/data/s3_file/FE/18January03FebCreative000'
			]
	training_file_name = '/data/s3_file/FE/18January03FebTrainingFile'
	generateCleanFile(files, training_file_name)
	trainModel(learning_rate, max_depth, training_file_name)

def __main__():
	# count the arguments
	if len(sys.argv) < 3:
		raise RuntimeError("Please provide the learning_rate and max_depth")
	startSteps(sys.argv[1], sys.argv[2])

#This is required to call the main function
if __name__ == "__main__":
	__main__()

