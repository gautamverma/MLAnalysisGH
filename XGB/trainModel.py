import sys
import boto3
import pickle
import logging
import datetime
import sagemaker

import numpy as np
from os import path
import pandas as pd

import utils as utils
import constants as const

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def trainXGBModel(learning_params, data_input):

	# Init a base Model
	xg_reg = {}

	YColumns = ['result']
	numericalCols = ['impressions', 'guarantee_percentage', 'container_id_label']
	#categoricalCols = [ 'component_name', 'slot_names', 'container_type', 'component_namespace',
	#					'component_display_name', 'customer_targeting', 'site']
	categoricalCols = [ 'component_name', 'slot_names', 'container_type', 'component_namespace',
						'component_display_name', 'customer_targeting', 'site', 'objective', 'intent']

	startOneHotIndex = len(numericalCols)
	columns_to_keep = YColumns + numericalCols + categoricalCols
	one_hot_encoder = buildOneHotEncoder(training_file_name, categoricalCols)
	logging.info('One hot encoder')

	# Convert it as it used for comparison 
	impression_count = int(impression_count)
	#Model present then load and predict
	if model_filename is not None and path.exists(model_filename):
		logging.info("Model file present. Skipping to predication::")
		xg_reg = pickle.load(open(model_filename, 'rb'))
		predict(training_file_name, one_hot_encoder, xg_reg, impression_count)
		return

	chunkcount = 1
	logging.info("Training for  " + data_input[const.IOBJECTIVE_KEY])
	logging.info("Training for total chunks : "+str(TRAIN_ITERATION))
	for chunk in pd.read_csv(training_file_name, chunksize=CHUNKSIZE):
		if(chunkcount%TRAIN_MOD_COUNT == 0):
			chunkcount = chunkcount + 1
			continue

		logging.info('Starting Training - '+str(chunkcount))
		chunk['result'] = chunk.apply (lambda row: label_result(row), axis=1)

		# Get only the columns to evaluate
		chunk = chunk[columns_to_keep + ['weblab']]

		# Fill All Categorical Missing Values
		chunk = utils.removeNaN(chunk, YColumns + numericalCols, NUMERIC_FILLER)
		chunk = utils.removeNaN(chunk, categoricalCols, CONSTANT_FILLER)

		# Get all rows where weblab is missing
		df_merged_set_test = chunk.where(chunk['weblab']=="missing").dropna()
		df_merged_set_test = df_merged_set_test[columns_to_keep]
		logging.info('Weblab Removed: Shape - '+str(df_merged_set_test.shape))

		INPUT = df_merged_set_test[numericalCols]

		ONEHOT = df_merged_set_test[categoricalCols]
		OUTPUT = df_merged_set_test[YColumns]

		logging.info(str(INPUT.columns))
		logging.info(str(ONEHOT.columns))

		one_hot_encoded = one_hot_encoder.transform(ONEHOT)
		logging.info('One hot encoding done')
		dataMatrix = xgb.DMatrix(np.column_stack((INPUT.iloc[:,1:], one_hot_encoded)), label=OUTPUT)

		if(chunkcount==1):
			xg_reg = xgb.train(learning_params, dataMatrix, 200)
		else:
			# Takes in the intially model and produces a better one
			xg_reg = xgb.train(learning_params, dataMatrix, 200, xgb_model=xg_reg)
		chunkcount = chunkcount + 1
		logging.info("Model saved "+str(xg_reg))

	return xg_reg
