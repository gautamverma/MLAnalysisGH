import sys
import pickle
import logging
import datetime

from os import path
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# This will only use the Label Encoder as on using the one hot encoding 
# we have to guarantee all the values in that one-hot-column has to present 
# in the test dataset also otherwise it will result in the feature mismatch bug

# TODO To resolve this we need put the columns in same order and intialize the old columns 
# We will do this in later in this we are training a simple XGB Model for the seven days data

# Batch size of 1M
CHUNKSIZE = 1000000
CONSTANT_FILLER = 'unknown_flag'

def mergeDataframe(df1, df2, column, joinType='inner'):
	if column is None:
		raise RuntimeError("Column can't be null. Please give the column value")
	return pd.merge(df1, df2, on=column, how=joinType);


def imputeMissingCols(df, numericCols, categoricalCols):
	logging.info("Starting the imputer for a dataframe")
	for col in numericCols:
		isNullPresent = pd.isnull(df.iloc[:,col]).any() 
		logging.info(isNullPresent)
		if isNullPresent:
			numericImputer = SimpleImputer(missing_values=np.nan, strategy='median')
			numericImputer.fit(df.iloc[:,col:col+1])	
			df.iloc[:,col:col+1] = numericImputer.transform(df.iloc[:,col:col+1])

	for col in categoricalCols:
		isNullPresent = pd.isnull(df.iloc[:,col]).any() 
		logging.info(isNullPresent)
		if isNullPresent:
			# Can't use the most_frequest in 10M+ Series as it takes very long time to update it
			categoricalImputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=CONSTANT_FILLER)
			categoricalImputer.fit(df.iloc[:,col:col+1])
			df.iloc[:,col:col+1] = categoricalImputer.transform(df.iloc[:,col:col+1])
	logging.info("Running of the imputer for a dataframe completed")		
	return df;

def cleanDataframe(df):
	datatypes = list(df.dtypes.iteritems())
	categoricalCols, numericCols = [], []
	for  i in range(len(datatypes)):
		str_dtype = str(datatypes[i][1])
		if str_dtype.startswith('int') or str_dtype.startswith('float'):
			numericCols.append(i)
		else:
			categoricalCols.append(i)
	return imputeMissingCols(df, numericCols, categoricalCols);

def loadDatasets(base_folder, cleanDframe):
	logging.info("Starting the loading of the datasets")

	PlACEMENTS__FILENAME = '3Hour1DecemberPMetadata000'
	CONTENT_FILENAME = '3Hour1DecemberCM000'
	PLACEMENT_PROPERTIES_FILENAME = '3Hour1DecemberPP000'

	# Make sure you download these file in the backed EC2 instance
	metrics_df1 = pd.read_csv(base_folder + PlACEMENTS__FILENAME, skiprows=0, header=None)
	metrics_df1.columns = ['frozen_placement_id', 'frozen_content_id', 'guarantee_percentage', 'created_by']
	# for null guarantee fill the explicitely 0
	if 'guarantee_percentage' in metrics_df1.columns:
		metrics_df1 = metrics_df1.replace(np.nan, 0)
	if cleanDframe:
		metrics_df1  = cleanDataframe(metrics_df1)

	metrics_df2 = pd.read_csv(base_folder + CONTENT_FILENAME, skiprows=0, header=None)
	metrics_df2.columns = ['frozen_content_id', 'component_name', 'component_namespace', 'created_by']
	if cleanDframe:
		metrics_df2  = cleanDataframe(metrics_df2)


	metrics_df3 = pd.read_csv(base_folder + PLACEMENT_PROPERTIES_FILENAME, skiprows=0, header=None)
	metrics_df3.columns  = ['frozen_placement_id', 'container_type', 'container_id', 
				'slot_names', 'site', 'start_date', 'end_date']

	if cleanDframe:
		metrics_df3  = cleanDataframe(metrics_df3)

	logging.info("Dataset cleaned "+str(cleanDframe)+" and loaded")
	return metrics_df1, metrics_df2, metrics_df3;

def trainModel(base_folder, clean, outputfile):

	chunkcount = 1
	cleanDframe = True if clean=='1' else False

	logging.info("Base folder:: clean dataframe "+base_folder+"::"+str(cleanDframe))
	df1, df2, df3 = loadDatasets(base_folder, cleanDframe)	

	metrics_file = base_folder + outputfile
	training_data_file = base_folder + '3Hour1DecemberPM000'
	for chunk in pd.read_csv(training_data_file, chunksize=CHUNKSIZE):
		logging.info("Start chunk Processing - " + str(chunkcount))
		chunk.columns = ['frozen_placement_id', 'impressions', 'metrics_hour']

		#format the timestamp columns
		chunk['metrics_hour'] = pd.to_datetime(chunk['metrics_hour'], format='%Y %m %d %H:%M:%S')
		chunk['metrics_hour'] = chunk['metrics_hour'].dt.tz_localize(None)
		
		df3['start_date'] =  pd.to_datetime(df3['start_date'], format='%Y %m %d %H:%M:%S')
		df3['end_date'] =  pd.to_datetime(df3['end_date'], format='%Y %m %d %H:%M:%S')

		df_merged_set = mergeDataframe(chunk, df1, 'frozen_placement_id')
		df_merged_set = mergeDataframe(df_merged_set, df2, 'frozen_content_id')
		df_merged_set = mergeDataframe(df_merged_set, df3, 'frozen_placement_id')
		
		logging.info("Merged dataframes")
		if not cleanDframe:
			df_merged_set = df_merged_set.dropna()

		# Generate the days and hour interval time gaps
		deltaTime = (df_merged_set['metrics_hour'] - df_merged_set['start_date']).dt
		df_merged_set['days_interval']  = deltaTime.days
		df_merged_set['hours_interval'] = deltaTime.total_seconds()/3600
		logging.info("Time intervals computed")

		columns_to_write = df_merged_set.columns
		if chunkcount == 1:
			df_merged_set.to_csv(metrics_file, columns=columns_to_write, index=False, mode='a')
		else:
			df_merged_set.to_csv(metrics_file, columns=columns_to_write, index=False, header=False, mode='a')
		chunkcount = chunkcount + 1 

def __main__():
	# count the arguments
	if len(sys.argv) < 4:
		raise RuntimeError("Please provode the base folder, cleanDataframe(0/1) and outputfile name")
	trainModel(sys.argv[1], sys.argv[2], sys.argv[3])

#This is required to call the main function
if __name__ == "__main__":
	__main__()