import datetime
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

def hyperParametersTuning(bucket, jsonprefix, base_folder):
    data_input = prepareInputData (bucket, jsonprefix, base_folder)
    start_steps (data_input, base_folder)

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