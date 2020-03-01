import sys
import pickle
import logging
from os import path

import boto3
import constants as const
import pandas as pd
import s3utils as s3utils
import sagemaker

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# It is only done for the container_id column
def label_column(df, baseFolder, column):
	map_file = baseFolder + column + "_map.dict"
	if path.exists(map_file):
		logging.info("Label unique hash file exists for col: " + column)
		logging.info("Filename is " + map_file)
		unique_container_id_hash = pickle.load(open(map_file, 'rb'))
		df['container_id_label'] = df.apply(lambda row: unique_container_id_hash[row[column]], axis=1)
		return df

	unique_container_id_list = df.container_id.unique().tolist()

	unique_container_id_hash = {}
	position = 1
	for val in unique_container_id_list:
		unique_container_id_hash[val] = position
		position = position + 1

	df['container_id_label'] = df.apply(lambda row: unique_container_id_hash[row[column]], axis=1)
	logging.info('Label done for ' + column)
	return df


# Give the file in order for loading
def loadAndMerge(data_input):
	# placement_metrics_file  {0}
	# placement_metadata_file {1}
	# content_metadata_file {2}
	# placement_properties_file {3}
	# creative_metadata_file {4}

	s3 = boto3.resource('s3')
	bucket = data_input[const.IBUCKET_KEY]
	prefix = data_input[const.IPREFIX_KEY]
	files = data_input[const.IFILES_KEY]
	base_folder = data_input[const.IFOLDER_KEY]

	file1 = s3utils.downloadFileFromS3(s3, bucket, prefix + files[0], base_folder + files[0])
	df1 = pd.read_csv(file1, skiprows=0, header=0)
	logging.info(df1.columns)

	file2 = s3utils.downloadFileFromS3(s3, bucket, prefix + files[1], base_folder + files[1])
	df2 = pd.read_csv(file2, skiprows=0, header=0)
	logging.info(df2.columns)

	df2[['guarantee_percentage']] = df2[['guarantee_percentage']].fillna(value=const.NUMERIC_FILLER)
	logging.info('Cleaned the placement metadata file');

	file3 = s3utils.downloadFileFromS3(s3, bucket, prefix + files[2], base_folder + files[2])
	df3 = pd.read_csv(file3, skiprows=0, header=0)
	logging.info(df3.columns)

	df3[['component_name']] = df3[['component_name']].fillna(value=const.CONSTANT_FILLER)
	df3[['component_namespace']] = df3[['component_namespace']].fillna(value=const.CONSTANT_FILLER)
	df3[['component_display_name']] = df3[['component_display_name']].fillna(value=const.CONSTANT_FILLER)
	logging.info("Clean the content metadata file")

	file4 = s3utils.downloadFileFromS3(s3, bucket, prefix + files[3], base_folder + files[3])
	df4 = pd.read_csv(file4, skiprows=0, header=0)
	logging.info(df4.columns)

	df4[['site']] = df4[['site']].fillna (value=const.CONSTANT_FILLER)
	df4[["language"]] = df4[['language']].fillna(value=const.CONSTANT_FILLER)
	df4[['slot_names']] = df4[['slot_names']].fillna (value=const.CONSTANT_FILLER)
	df4[['container_id']] = df4[['container_id']].fillna(value=const.CONSTANT_FILLER)
	df4[['container_type']] = df4[['container_type']].fillna (value=const.CONSTANT_FILLER)
	df4[['customer_targeting']] = df4[['customer_targeting']].fillna(value=const.ALL_CONSUMER)

	# Generate the unique set and map values
	df4 = label_column(df4, base_folder, 'container_id')
	logging.info("Clean the container_id column of placement properties file")

	file5 = s3utils.downloadFileFromS3(s3, bucket, prefix + files[4], base_folder + files[4])
	df5 = pd.read_csv(file5, skiprows=0, header=0)
	logging.info(df5.columns)

	# Creative Columns
	df5[['intent']] = df5[['intent']].fillna(value=const.CONSTANT_FILLER)
	df5[['objective']] = df5[['objective']].fillna(value=const.CONSTANT_FILLER)
	logging.info('Creative Columns Cleaned')

	logging.info('File Loaded')

	df_merged_set = pd.merge(df1, df4, on='frozen_placement_id', how='inner')
	df_merged_set = pd.merge(df_merged_set, df2, on='frozen_placement_id', how='inner')
	df_merged_set = pd.merge(df_merged_set, df3, on='frozen_content_id', how='inner')
	df_merged_set = pd.merge(df_merged_set, df5, on='creative_id', how='inner')
	logging.info('File merged');
	return df_merged_set


def generateCleanFile(data_input):
	training_file = data_input[const.ITRAINING_FP]
	if path.exists(training_file):
		logging.info("Training file is already present")
		return

	df_merged_set = loadAndMerge(data_input)
	logging.info("Loading and dataset merged. Display columns")
	logging.info(df_merged_set.columns)
	logging.info("Dataframe Shape " + str(df_merged_set.shape))

	df_merged_set.to_csv(training_file, index=False, encoding='utf-8')
	logging.info('File Created')
