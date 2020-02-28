import sys
import json

import pandas as pd

import utils as utils
import constants as const
import s3utils as s3utils

from trainModel import trainXGBModel
from predication import predictXGBModel

def prepareInputData(bucket, jsonprefix, base_folder):
	filepath = s3utils.downloadFileFromS3(bucket, jsonprefix, base_folder + 'input.json')

	data_input = {}
	with open(filepath, "r") as input_file:
		data = json.load(input_file)
		data_input[const.IBUCKET_KEY] = bucket
		data_input[const.IRESULT_PREFIX_KEY] = data[const.IRESULT_PREFIX_KEY]

		data_input[const.IFILES_KEY] =  data[const.IFILES_KEY]
		data_input[const.ISTARTEGY_KEY] = data[const.ISTARTEGY_KEY]

 		data_input[const.INUMERICAL_COLS] = data[const.INUMERICAL_COLS]
		data_input[const.ICATEGORICAL_COLS] = data[const.ICATEGORICAL_COLS]

		data_input[const.IFOLDER_KEY] = base_folder
		data_input[const.ITRAINING_FP] = base_folder + '/training_file'
		data_input[const.IMODEL_FN] = data[const.IMODEL_FN]
		data_input[const.IMODEL_FP] = base_folder + data[const.IMODEL_FN]


	return data_input

def start_steps(bucket, jsonprefix, base_folder):
	data_input = prepareInputData(sys.argv[1], sys.argv[2], sys.argv[3])
	xgb_model = trainXGBModel(data_input)
	# Save Model on the disk
	utils.saveDataOnDisk(xgb_model, data_input[const.IMODEL_FP])
	s3utils.uploadFiletoS3(bucket, data_input[const.IRESULT_PREFIX_KEY] + '/' + data_input[const.IMODEL_FN], data_input[const.IMODEL_FP])

	# Predict and save the accuracy per chunk values
	accuracy_scope_filename, accuracy_scope_filepath = predictXGBModel(data_input, xgb_model)
	s3utils.uploadFiletoS3(data_input[const.IBUCKET_KEY], data_input[const.IRESULT_PREFIX_KEY] +'/'+ accuracy_scope_filename, accuracy_scope_filepath)

def __main__():
	# count the arguments
	if len(sys.argv) < 3:
		raise RuntimeError("Please provide the bucket,prefix to load the input data config and base folder")

	#Validations
	if sys.argv[3].endswith('/'):
		raise RuntimeError('Please add base folder ending with /')
	start_steps(bucket=sys.argv[1], prefix=sys.argv[2], base_folder=sys.argv[3])

#This is required to call the main function
if __name__ == "__main__":
	__main__()