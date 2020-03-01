import sys
import pickle
import logging

from os import path

import numpy as np
import pandas as pd
import xgboost as xgb

import utils as utils
import constants as const

from enumclasses import MLFunction
from resultFunctions import callFunctionByName

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def trainXGBModel(data_input):
	# Init a base Model
	xg_reg = {}

	# Model present then load and return
	model_filepath = data_input[const.IMODEL_FP]
	if model_filepath is not None and path.exists(model_filepath):
		logging.info("Model file present. Skipping to predication::")
		xg_reg = pickle.load(open(model_filepath, 'rb'))
		return xg_reg

	# Predication will be always on 1 result col
	YColumns = [data_input[const.IRESULT_COL_KEY]]
	numericalCols = data_input[const.INUMERICAL_COLS]
	categoricalCols = data_input[const.ICATEGORICAL_COLS]

	columns_to_keep = YColumns + numericalCols + categoricalCols
	one_hot_encoder, shapeTuple = utils.buildOneHotEncoder(data_input[const.ITRAINING_FP] , categoricalCols)
	logging.info('One hot encoder is ready')

	chunkcount = 1
	TOTAL_CHUNK_COUNT = shapeTuple[0] / data_input[const.ICHUNKSIZE_KEY]

	logging.info("Training for  " + data_input[const.IOBJECTIVE_KEY])
	logging.info("Training using stragegy : "+str(data_input[const.ISTARTEGY_KEY]))
	for chunk in pd.read_csv(data_input[const.ITRAINING_FP], chunksize=data_input[const.ICHUNKSIZE_KEY]):
		if not utils.useChunk(data_input[const.ISTARTEGY_KEY], MLFunction.Train, chunkcount, TOTAL_CHUNK_COUNT):
			chunkcount = chunkcount + 1
			continue

		logging.info('Starting Training - '+str(chunkcount))
		chunk[data_input[const.IRESULT_COL_KEY]] = chunk.apply (lambda row: callFunctionByName(row, data_input[const.IRESULT_FUNCTION]), axis=1)

		# Get only the columns to evaluate
		chunk = chunk[columns_to_keep + ['weblab']]

		# Get all rows where weblab is missing
		df_merged_set_test = chunk.where(chunk['weblab']=="missing").dropna()
		df_merged_set_test = df_merged_set_test[columns_to_keep]
		logging.info('Weblab Removed: Shape - '+str(df_merged_set_test.shape))

		INPUT = df_merged_set_test[numericalCols]

		ONEHOT = df_merged_set_test[categoricalCols]
		OUTPUT = df_merged_set_test[YColumns]

		one_hot_encoded = one_hot_encoder.transform(ONEHOT)
		logging.info('One hot encoding done')
		dataMatrix = xgb.DMatrix(np.column_stack((INPUT.iloc[:,1:], one_hot_encoded)), label=OUTPUT)

		if(chunkcount==1):
			xg_reg = xgb.train(data_input[const.IPARAMS_KEY], dataMatrix, data_input[const.ITRAIN_ITERATIONS])
		else:
			# Takes in the intially model and produces a better one
			xg_reg = xgb.train(data_input[const.IPARAMS_KEY], dataMatrix, data_input[const.ITRAIN_ITERATIONS], xgb_model=xg_reg)
		chunkcount = chunkcount + 1
		logging.info("Model saved for "+chunkcount)
		if chunkcount == 2:
			break
	return xg_reg
