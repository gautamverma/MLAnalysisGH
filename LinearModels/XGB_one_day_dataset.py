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

# Batch size of 10000
CHUNKSIZE = 10000
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

# Load the Labels Vocabulary in One Hot Encoder
def loadCategorialList(base_folder, columnNm):
	map_file = base_folder + columnNm + "_map.dict"
	if not path.exists(map_file):
		raise RuntimeError("Map file missing for "+ columnNm + " name")

	column_dict_file = open(map_file, "rb")
	column_dict = pickle.load(column_dict_file)
	column_series = pd.Series(column_dict)
	return list(column_series.index)

def saveModel(xg_reg, learning_rate_val, max_depth_val):
	filename =  '/data/models/XGB_MODEL_{}_{}_{}.sav'
	filename  = filename.format(learning_rate_val, max_depth_val, int(datetime.datetime.now().timestamp())) 
	pickle.dump(xg_reg, open(filename, 'wb'))

	logging.info("training complete and model is saved")

def loadModel(learning_rate_val, max_depth_val):
	# Classifier Declared 
	xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = learning_rate_val, 
                         max_depth = max_depth_val, alpha = 5, n_estimators = 10)

def trainModel(learning_rate_val, max_depth_val, base_folder, clean):

	chunkcount = 1
	cleanDframe = True if clean=='1' else False

	logging.info("Base folder:: clean dataframe "+base_folder+"::"+str(cleanDframe))
	df1, df2, df3 = loadDatasets(base_folder, cleanDframe)
	
	xg_reg = loadModel(learning_rate_val, max_depth_val)

	#Load the categorical columns for faster filling in between
	categoricalCols = [ 'slot_names', 'container_type', 'component_name', 'component_namespace', 'site']
	categoryLists = []
	for col in categoricalCols:
		categoryLists.append(loadCategorialList(base_folder, col))

	one_hot_encoder = OneHotEncoder(categories=categoryLists, handle_unknown='ignore', sparse=False)	

	training_data_file = base_folder + '3Hour1DecemberPM000'
	for chunk in pd.read_csv(training_data_file, chunksize=CHUNKSIZE):
		logging.info("Start chunk Processing - " + str(chunkcount))
		chunkcount = chunkcount + 1 
		chunk.columns = ['frozen_placement_id', 'impressions', 'metrics_hour']

		#format the timestamp columns
		chunk['metrics_hour'] = pd.to_datetime(chunk['metrics_hour'], format='%Y %m %d %H:%M:%S')
		chunk['metrics_hour'] = chunk['metrics_hour'].dt.tz_localize(None)
		
		df3['start_date'] =  pd.to_datetime(df3['start_date'], format='%Y %m %d %H:%M:%S')
		df3['end_date'] =  pd.to_datetime(df3['end_date'], format='%Y %m %d %H:%M:%S')

		df_merged_set = mergeDataframe(chunk, df1, 'frozen_placement_id')
		logging.info("Merged df1")
	
		df_merged_set = mergeDataframe(df_merged_set, df2, 'frozen_content_id')
		logging.info("Merged df2")
	
		df_merged_set = mergeDataframe(df_merged_set, df3, 'frozen_placement_id')
		logging.info("Merged df3")
		if not cleanDframe:
			df_merged_set = df_merged_set.dropna()

		# Generate the days and hour interval time gaps
		deltaTime = (df_merged_set['metrics_hour'] - df_merged_set['start_date']).dt
		df_merged_set['days_interval']  = deltaTime.days
		df_merged_set['hours_interval'] = deltaTime.total_seconds()/3600
		logging.info("Time intervals computed")
		
		YColumns = ['impressions']
		numericCols = ['guarantee_percentage', 'days_interval', 'hours_interval']
		columns_to_keep = YColumns + categoricalCols + numericCols 
		df_merged_set = df_merged_set[columns_to_keep]	
		
		nLength = len(numericCols)
		cLength = len(categoricalCols)
		X1, X2, Y = df_merged_set.iloc[:,1:cLength+1], df_merged_set.iloc[:,cLength+1:cLength+nLength+1], df_merged_set.iloc[:,0]
		
		one_hot_encoder.fit(X1)
		one_hot_encoded = one_hot_encoder.transform(X1)

		logging.info(str(X2.size) + " : "+str(one_hot_encoded.size))

		# Drop the ctegorical columns 	
		dataMatrix = xgb.DMatrix(np.concatenate((X2, one_hot_encoded), axis=1), label=Y.to_numpy())
		xgb.train({}, dataMatrix, 1, xgb_model=xg_reg)
	logging.info(xg_reg)
	saveModel(xg_reg, learning_rate_val, max_depth_val)

def __main__():
	# count the arguments
	if len(sys.argv) < 5:
		raise RuntimeError("Please provode the learning_rate, max_depth, base folder and cleanDataframe(0/1)")
	trainModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

#This is required to call the main function
if __name__ == "__main__":
	__main__()