import logging
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append('..')

from utility import constants
from utility import utils
from resultFunctions import callFunctionByName

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def trainXGBModel(data_input, training_filepath):
	# Init a variable to keep the best Model
	xgb_model = {}
	otherValues = {
		'accuracy_value': 0
	}

	# Initialize the range to test the different parameters
	learning_rate_vals = np.arange (0.1, 0.5, 0.02)
	max_depth_vals = np.arange (8, 24, 1)
	n_estimators_vals = np.arange (100, 500, 100)

	# Predication will be always on 1 result col
	YColumns = [data_input[constants.IRESULT_COL_KEY]]
	numericalCols = data_input[constants.INUMERICAL_COLS]
	categoricalCols = data_input[constants.ICATEGORICAL_COLS]

	columns_to_keep = YColumns + numericalCols + categoricalCols
	one_hot_encoder, shapeTuple = utils.buildOneHotEncoder(training_filepath , categoricalCols)

	df = pd.read_csv(training_filepath, skiprows=0, header=0)
	df[data_input[constants.IRESULT_COL_KEY]] = df.apply(
		lambda row: callFunctionByName (row, data_input[constants.IRESULT_FUNCTION]), axis=1)

	df = df[columns_to_keep]
	Y, X = df.iloc[:, 0], df.iloc[:, 1:]

	learning_parameters = data_input[constants.IPARAMS_KEY]
	for learning_rate in learning_rate_vals:
		for max_depth in max_depth_vals:
			for n_estimators in n_estimators_vals:
				learning_parameters['max_depth'] = max_depth
				learning_parameters['n_estimators'] = n_estimators
				learning_parameters['learning_rate'] = learning_rate
				data_input[constants.IPARAMS_KEY] = learning_parameters

				X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size=0.2, random_state=123)

				numeric_data = X_train.iloc[:,0:len(numericalCols)]
				one_hot_encoded = one_hot_encoder.transform(X_train.iloc[:,len(numericalCols):])
				d_train = xgb.DMatrix (np.column_stack ((numeric_data, one_hot_encoded)), label=y_train)

				xg_reg = xgb.train(data_input[constants.IPARAMS_KEY], d_train, data_input[constants.ITRAIN_ITERATIONS])

				numeric_data = X_test.iloc[:,0:len(numericalCols)]
				one_hot_encoded = one_hot_encoder.transform(X_test.iloc[:,len(numericalCols):])
				d_test = xgb.DMatrix(np.column_stack ((numeric_data, one_hot_encoded)))
				preds = xg_reg.predict(d_test)

				accuracy = accuracy_score(y_test, np.around(preds))
				matrix = confusion_matrix(y_test, np.around (preds))

				logging.info(str(data_input[constants.IPARAMS_KEY]))
				logging.info('Confusion Matrix : ')
				logging.info(str(matrix))
				logging.info('Accuracy Score : ' + str (accuracy))
				logging.info(str(classification_report (y_test, np.around (preds))))

				if(otherValues['accuracy_value']<accuracy):
					xgb_model = xg_reg
					otherValues['accuracy_value'] = accuracy
					otherValues['Confusion_Matrix'] = matrix
					otherValues['Classification_report'] = classification_report(y_test, np.around (preds))
			logging.info("\n")
		logging.info('\n')
	logging.info('\n')

	logging.info('Best Model values')
	logging.info('')
	return xgb_model