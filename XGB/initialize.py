import json
import sys
import boto3
import logging
import datetime

import utils as utils
import constants as const
import s3utils as s3utils

from data_filters import filterProdEnviroment
from data_filters import filterNonMarketingData

from trainModel import trainXGBModel
from predication import predictXGBModel
from loadAndCleanFile import generateCleanFile

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def prepareInputData(bucket, jsonprefix, base_folder):
    s3 = boto3.resource ('s3')
    filepath = s3utils.downloadfilefroms3(s3, bucket, jsonprefix, base_folder + 'input.json')

    data_input = {}
    with open(filepath, "r") as input_file:
        data = json.load(input_file)
        logging.info (data)
        data_input[const.IBUCKET_KEY] = data[const.IBUCKET_KEY]
        data_input[const.IPREFIX_KEY] = data[const.IPREFIX_KEY]

        data_input[const.IFILES_KEY] = data[const.IFILES_KEY]
        data_input[const.ISTARTEGY_KEY] = data[const.ISTARTEGY_KEY]

        data_input[const.IMULTIPLE_METRICS_FILES] = data[const.IMULTIPLE_METRICS_FILES]

        data_input[const.IRESULT_COL_KEY] = data[const.IRESULT_COL_KEY]
        data_input[const.IRESULT_FUNCTION] = data[const.IRESULT_FUNCTION]
        data_input[const.INUMERICAL_COLS] = data[const.INUMERICAL_COLS]
        data_input[const.ICATEGORICAL_COLS] = data[const.ICATEGORICAL_COLS]

        data_input[const.IFOLDER_KEY] = base_folder
        data_input[const.ITRAINING_FP] = base_folder + 'training_file'
        data_input[const.IMULTIPLE_TRAINING_FILE] = base_folder + 'multi_training_file'

        data_input[const.IMODEL_FN] = data[const.IMODEL_FN]
        data_input[const.IMODEL_FP] = base_folder + data[const.IMODEL_FN]

        data_input[const.IPARAMS_KEY] = data[const.IPARAMS_KEY]
        data_input[const.ICHUNKSIZE_KEY] = data[const.ICHUNKSIZE_KEY]
        data_input[const.ITRAIN_ITERATIONS] = data[const.ITRAIN_ITERATIONS]
        data_input[const.IOBJECTIVE_KEY] = data[const.IOBJECTIVE_KEY]

    return data_input

def buildPredicationModel(data_input, training_file, s3_prefix):
    xgb_model = trainXGBModel (data_input, training_file)

    # Save Model on the disk
    utils.saveDataOnDisk (xgb_model, data_input[const.IMODEL_FP])
    s3utils.uploadFiletoS3 (data_input[const.IBUCKET_KEY], s3_prefix + data_input[const.IMODEL_FN],
                            data_input[const.IMODEL_FP])

    # Predict and save the accuracy per chunk values
    accuracy_scope_filename, accuracy_scope_filepath = predictXGBModel (data_input, training_file,  xgb_model)
    s3utils.uploadFiletoS3 (data_input[const.IBUCKET_KEY],
                            s3_prefix + accuracy_scope_filename,
                            accuracy_scope_filepath)

def start_steps(bucket, jsonprefix, base_folder):
    data_input = prepareInputData(bucket, jsonprefix, base_folder)
    generateCleanFile(data_input)

    utils.logBreak()
    data_input[const.IFILE_PREFIX] = 'full_file';
    timestamp_value = int (datetime.datetime.now ().timestamp ())
    buildPredicationModel(data_input, data_input[const.ITRAINING_FP], data_input[const.IPREFIX_KEY] + str (timestamp_value) + "_full/")

    utils.logBreak()
    data_input = filterProdEnviroment(data_input, data_input[const.ITRAINING_FP])
    timestamp_value = int (datetime.datetime.now ().timestamp ())
    buildPredicationModel(data_input, data_input[const.PROD_ENVIROMENT_FILTERED_FILE], data_input[const.IPREFIX_KEY] + str (timestamp_value) + "_ProdFiltered/")

    utils.logBreak()
    data_input = filterNonMarketingData(data_input, data_input[const.ITRAINING_FP])
    timestamp_value = int (datetime.datetime.now ().timestamp ())
    buildPredicationModel(data_input, data_input[const.NON_MARKETING_FILTERED_FILE], data_input[const.IPREFIX_KEY] + str (timestamp_value) + "_NonMarketingFiltered/")

def __main__():
    # count the arguments
    if len(sys.argv) < 3:
        raise RuntimeError("Please provide the bucket,prefix to load the input data config and base folder")

    logging.info("Parameters ::")
    logging.info(sys.argv)
    # Validations
    if not sys.argv[3].endswith('/'):
        raise RuntimeError('Please add base folder ending with /')
    start_steps(bucket=sys.argv[1], jsonprefix=sys.argv[2], base_folder=sys.argv[3])

# This is required to call the main function
if __name__ == "__main__":
    __main__()

#Example Command
#nohup python3 -u initializeSingleFile.py deep-learning-fe-datasets
# january/15Jan15Feb/input.json /home/ec2-user/SageMaker/FE-15Jan15Feb-dataset/ > FE-BinaryRun1March &