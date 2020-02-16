import sagemaker

import os
import boto3
import pickle
import datetime

from os import path
import pandas as pd
import numpy as np
import seaborn as sns

import sagemaker.xgboost as xgb

from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner 
 
 # Batch size of 200000
CHUNKSIZE = 200000
CONSTANT_FILLER = 'missing'
NUMERIC_FILLER = 0
ALL_CONSUMER    = 'allCustomer'

region = boto3.Session().region_name    
smclient = boto3.Session().client('sagemaker')

# Default role and bucket
role = sagemaker.get_execution_role()
bucket = sagemaker.Session().default_bucket()

# Download the file
input_bucket = 'sagemaker-campaign-predication-input'
prefix = 'FE-18Jan-3FebTrainingFile/18January03FebTrainingFile'
training_filename='training_file'

s3 = boto3.resource('s3')
if(path.exists(training_filename)):
    print('file present')
else:
    s3.Bucket(input_bucket).download_file(prefix, training_filename)

def label_result(row):
	if(row['impressions']<10):
		return 1
	elif(row['impressions']<100):
		return 2
	elif(row['impressions']<1000):
		return 3
	elif(row['impressions']<10000):
		return 4
	elif(row['impressions']<100000):
		return 5
	return 0

def removeNaN(df, categoricalCols, defValue):
	# Replace any NaN values
	for col in categoricalCols:
		df[[col]] = df[[col]].fillna(value=defValue)
	return df

# Build the One hot encoder using all data
def buildOneHotEncoder(training_file_name, categoricalCols):
	# using a global keyword 
	global TRAIN_ITERATION

	one_hot_encoder = OneHotEncoder(sparse=False)
	df = pd.read_csv(training_file_name, skiprows=0, header=0)

	TOTAL_CHUNK_COUNT = df.shape[0]/CHUNKSIZE 
	TRAIN_ITERATION = int((75*TOTAL_CHUNK_COUNT)/100)

	print("ChunkSize: Iterations ::" +str(TOTAL_CHUNK_COUNT)+ " : " +str(TRAIN_ITERATION))
	df = df[categoricalCols]
	df = removeNaN(df, categoricalCols, CONSTANT_FILLER)
	print(str(df.columns))
	one_hot_encoder.fit(df)
	return one_hot_encoder

def buildCleanFile():

	YColumns = ['result']
	numericalCols = ['guarantee_percentage', 'container_id_label']
	categoricalCols = [ 'component_name', 'slot_names', 'container_type', 'component_namespace',
						'component_display_name', 'customer_targeting', 'site']

	startOneHotIndex = len(numericalCols)
	columns_to_keep = YColumns + numericalCols + categoricalCols
	one_hot_encoder = buildOneHotEncoder(training_file_name, categoricalCols)

	LARGECHUNK = 3000000
	for chunk in pd.read_csv(training_filename, chunksize=LARGECHUNK):

		# Add the result column
		chunk['result'] = chunk.apply (lambda row: label_result(row), axis=1)

		# Get only the columns to evaluate
		chunk = chunk[columns_to_keep + ['weblab']]
		
		# Fill All Categorical Missing Values
		chunk = removeNaN(chunk, YColumns + numericalCols, NUMERIC_FILLER)
		chunk = removeNaN(chunk, categoricalCols, CONSTANT_FILLER)

		df_merged_set_test = chunk.where(chunk['weblab']=="missing").dropna()
		df_merged_set_test = df_merged_set_test[columns_to_keep]

		INPUT = df_merged_set_test[numericalCols]
		# guarantee_percentage nan replaced by missing so change back
		INPUT.replace(CONSTANT_FILLER, NUMERIC_FILLER, inplace=True)

		ONEHOT = df_merged_set_test[categoricalCols]
		OUTPUT = df_merged_set_test[YColumns]

		one_hot_encoded = one_hot_encoder.transform(ONEHOT)





def saveModel(xg_reg, learning_params, columns_to_keep):
	# Dump the model in the Notebook Instance and upload it to S3
	model_filename =  'XGB_MODEL_impression-timestamp{}.sav'
	timestamp_value = int(datetime.datetime.now().timestamp())
	model_filename  = model_filename.format(timestamp_value) 
	pickle.dump(xg_reg, open(model_filename, 'wb'))
	
	column_filename =  '/data/s3_file/models/XGB_MODEL_COLUMN_{}.sav'
	column_filename = column_filename.format(timestamp_value)
	pickle.dump(columns_to_keep, open(column_filename, 'wb'))

	s3.Bucket(input_bucket).upload_file(model_filename, 'results/'+model_filename)
	s3.Bucket(input_bucket).upload_file(column_filename, 'results/'+column_filename)

def trainModel():

	learning_params = {
	    'objective' : 'multi:softmax',
	    'colsample_bytree' : 0.3,
	    'learning_rate' : 0.3, 
	    'max_depth' : 16,
	    'alpha' : 5,
	    'num_class': 6,
	    'n_estimators' : 200
	}

	# Init a base Model
	xg_reg = {}

	YColumns = ['result']
	numericalCols = ['impressions', 'guarantee_percentage', 'container_id_label']
	categoricalCols = [ 'component_name', 'slot_names', 'container_type', 'component_namespace',
						'component_display_name', 'customer_targeting', 'site']

	startOneHotIndex = len(numericalCols)
	columns_to_keep = YColumns + numericalCols + categoricalCols
	one_hot_encoder = buildOneHotEncoder(training_file_name, categoricalCols)
	print('One hot encoder')

	chunkcount = 1
	print("Training for placements impressions < "+str(impression_count))
	print("Training for total chunks : "+str(TRAIN_ITERATION))
	for chunk in pd.read_csv(training_file_name, chunksize=CHUNKSIZE):
		# Train on a part of dataset and predict on other
		if(chunkcount>TRAIN_ITERATION):
			break

		print('Starting Training - '+str(chunkcount))
		chunk['result'] = chunk.apply (lambda row: label_result(row), axis=1)

		# Get only the columns to evaluate
		chunk = chunk[columns_to_keep + ['weblab']]

		# Fill All Categorical Missing Values
		chunk = removeNaN(chunk, YColumns + numericalCols, NUMERIC_FILLER)
		chunk = removeNaN(chunk, categoricalCols, CONSTANT_FILLER)

		# Get all rows where weblab is missing
		df_merged_set_test = chunk.where(chunk['weblab']=="missing").dropna()
		df_merged_set_test = df_merged_set_test[columns_to_keep]
		print('Weblab Removed: Shape - '+str(df_merged_set_test.shape))

		INPUT = df_merged_set_test[numericalCols]
		# guarantee_percentage nan replaced by missing so change back
		INPUT.replace(CONSTANT_FILLER, NUMERIC_FILLER, inplace=True)

		ONEHOT = df_merged_set_test[categoricalCols]
		OUTPUT = df_merged_set_test[YColumns]

		print(str(INPUT.columns))
		print(str(ONEHOT.columns))

		one_hot_encoded = one_hot_encoder.transform(ONEHOT)
		print('One hot encoding done')
		dataMatrix = xgb.DMatrix(np.column_stack((INPUT.iloc[:,1:], one_hot_encoded)), label=OUTPUT)

		if(chunkcount==1):
			xg_reg = xgb.train(learning_params, dataMatrix, 200)
		else:
			# Takes in the intially model and produces a better one
			xg_reg = xgb.train(learning_params, dataMatrix, 200, xgb_model=xg_reg)
		chunkcount = chunkcount + 1
		print("Model saved "+str(xg_reg))

	saveModel(xg_reg, learning_params, columns_to_keep)
	return
