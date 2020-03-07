import sys
import pickle
import logging
from os import path

import boto3
import constants as const
import pandas as pd
import s3utils as s3utils

from loadAndCleanFile import loadAndMerge

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def generateCleanFile(data_input):
	training_file = data_input[const.IMULTIPLE_TRAINING_FILE]
	if path.exists(training_file):
		logging.info("Multiple Training file is already present")
		return

	for metric_file in data_input[const.IMULTIPLE_METRICS_FILES]:
        data_input[const.IFILES_KEY][0] = metric_file
        df_merged_set = loadAndMerge(data_input)

        logging.info("Loading and dataset merged. Display columns")
        logging.info(df_merged_set.columns)
        logging.info("Dataframe Shape " + str(df_merged_set.shape))

        df_merged_set.to_csv(training_file, index=False, encoding='utf-8', mode='a')

	logging.info('File Created')

