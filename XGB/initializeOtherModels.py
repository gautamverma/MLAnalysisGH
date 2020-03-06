import sys
import datetime
import logging
import sys

import constants as const
import data_filters as filters
from initialize import buildPredicationModel
from initialize import prepareInputData
from loadAndCleanFile import generateCleanFile

import utils as utils

def start_steps(bucket, jsonprefix, base_folder):
    data_input = prepareInputData(bucket, jsonprefix, base_folder)
    generateCleanFile(data_input)

    utils.logBreak()
    data_input = filters.filterProdEnviroment(data_input)
    timestamp_value = int (datetime.datetime.now ().timestamp ())
    buildPredicationModel(data_input, data_input[const.PROD_ENVIROMENT_FILTERED_FILE], data_input[const.IPREFIX_KEY] + str (timestamp_value) + "_ProdFiltered/")

    utils.logBreak()
    data_input = filters.filterNonMarketingData(data_input)
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



