import datetime
import logging
import numpy as np

# To prepare the other modules
import sys
sys.path.append('..')

from utility import constants
from utility import utils
from initialize import prepareInputData
from TrainAndTest import trainXGBModel


def start_steps(data_input, base_folder):

    xgb_model = trainXGBModel(data_input, data_input[constants.ITRAINING_FP])

    # Model present then save the timeprefix and save it
    data_input[constants.IMODEL_FN] = str(int(datetime.datetime.now().timestamp())) + data_input[constants.IMODEL_FN]
    data_input[constants.IMODEL_FP] = base_folder + data_input[constants.IMODEL_FN]
    utils.saveDataOnDisk (xgb_model, data_input[constants.IMODEL_FP])
    return

def hyperParametersTuning(bucket, jsonprefix, base_folder):
    data_input = prepareInputData (bucket, jsonprefix, base_folder)

    learning_parameters = {
        "objective": "binary:logistic",
        "colsample_bytree": 0.3,
        "learning_rate": 0.1,
        "max_depth": 8,
        "alpha": 5,
        "n_estimators": 50
    }

    learning_rate_vals = np.arange(0.1, 0.5, 0.02)
    max_depth_vals     = np.arange(8, 24, 1)
    n_estimators_vals  = np.arange(50, 500, 10)

    for learning_rate in learning_rate_vals:
        for max_depth in max_depth_vals:
            for n_estimators in n_estimators_vals:
                learning_parameters.max_depth = max_depth
                learning_parameters.n_estimators = n_estimators
                learning_parameters.learning_rate = learning_rate
                data_input[constants.IPARAMS_KEY] = learning_parameters
                start_steps(data_input, base_folder)
    return

def __main__():
    # count the arguments
    if len(sys.argv) < 3:
        raise RuntimeError("Please provide the bucket,prefix to load the input data config and base folder")

    # Validations
    if not sys.argv[3].endswith('/'):
        raise RuntimeError('Please add base folder ending with /')
    hyperParametersTuning(bucket=sys.argv[1], jsonprefix=sys.argv[2], base_folder=sys.argv[3])

# This is required to call the main function
if __name__ == "__main__":
    __main__()