import sys
import pickle
import logging
import datetime

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

# Batch size of 5M
CHUNKSIZE = 5000000

def mergeDataframe(df1, df2, column, joinType='inner'):
	if column is not None:
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
			categoricalImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
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


def loadDatasets(cleanDframe):
	logging.info("Starting the loading of the datasets")
	PlACEMENTS__FILENAME = '3_10_files/3_10NovemberDS_Placementfeature000'
	CONTENT_FILENAME = '3_10_files/3_10NovemberDS_Contentfeature000'
	RESOURCEBUNDLE_FILENAME = '3_10_files/3_10NovemberDS_Bundlefeature000'
	PLACEMENT_PROPERTIES_FILENAME = '3_10_files/3_10November0S_7feature000'

	# Make sure you download these file in the backed EC2 instance
	metrics_df1 = pd.read_csv('/data/s3_file/'+PlACEMENTS__FILENAME, skiprows=0, header=None)
	metrics_df1.columns = ['frozen_placement_id', 'frozen_content_id', 'guarantee_percentage', 'created_by']
	# for null guarantee fill the explicitely 0
	if 'guarantee_percentage' in metrics_df1.columns:
		metrics_df1 = metrics_df1.replace(np.nan, 0)
	if cleanDframe:
		metrics_df1  = cleanDataframe(metrics_df1)

	metrics_df2 = pd.read_csv('/data/s3_file/'+CONTENT_FILENAME, skiprows=0, header=None)
	metrics_df2.columns = ['frozen_content_id', 'component_name', 'component_namespace', 'created_by']
	if cleanDframe:
		metrics_df2  = cleanDataframe(metrics_df2)

	metrics_df3 = pd.read_csv('/data/s3_file/'+RESOURCEBUNDLE_FILENAME, skiprows=0, header=None)
	metrics_df3.columns = ['frozen_content_id', 'language_code']
	if cleanDframe:
		metrics_df3  = cleanDataframe(metrics_df3)

	metrics_df4 = pd.read_csv('/data/s3_file/'+PLACEMENT_PROPERTIES_FILENAME, skiprows=0, header=None)
	metrics_df4.columns = ['frozen_placement_id', 'container_type', 'container_id', 'slot_names', 'merchant_id', 'site', 'weblab', 'bullseye', 'is_recognized', 'parent_browse_nodes', 'store_names','start_date', 'end_date']
	
	# Removing the some misssing datasets to reduce running time
	metrics_df4  = metrics_df4[['frozen_placement_id', 'container_type', 'container_id', 
				'slot_names', 'merchant_id', 'site','start_date', 'end_date']]

	if cleanDframe:
		metrics_df4  = cleanDataframe(metrics_df4)

	logging.info("Dataset cleaned "+str(cleanDframe)+" and loaded")
	return metrics_df1, metrics_df2, metrics_df3, metrics_df4;

def labelCategoryColumns(df, cols):
	for col in cols:
		df.loc[:,col] = label_encoder.fit_transform(df.loc[:,col]).astype('int64')
	return df;	

def generateCategoricalData(df, categoricalCols):
	# It generates the numrical values for the categories where nan is assigned -1 and 
	# others start with 0 
	for col in categoricalCols:
		df[col] = pd.Categorical(df[col])
		df[col] = df[col].cat.codes
	return df

def loadCategorialList(base_folder, columnNm):
	map_file = base_folder + columnNm + "_map.dict"
	if not path.exists(map_file):
		raise RuntimeError("Map file missing for "+ columnNm + " name")

	column_dict_file = open(map_file, "rb")
	column_dict = pickle.load(column_dict_file)
	column_series = pd.Series(column_dict)
	return column_series, column_series.tolist()

def fillLabelFromFile(df, columnNm, column_series):
	df[columnNm +"_label"] = column_series[df[columnNm]]
	df = df.drop([columnNm], axis=1)
	df.rename(columns={columnNm + "_label": columnNm}, inplace=True)
	return  df


def saveModel(xg_reg, label_encoder, learning_rate_val, max_depth_val):
	filename =  '/data/models/XGB_MODEL_{}_{}_{}.sav'
	filename  = filename.format(learning_rate_val, max_depth_val, int(datetime.datetime.now().timestamp())) 
	pickle.dump(xg_reg, open(filename, 'wb'))

	filename =  '/data/models/XGB_LABEL_{}_{}_{}.sav'
	filename  = filename.format(learning_rate_val, max_depth_val, int(datetime.datetime.now().timestamp())) 
	pickle.dump(label_encoder, open(filename, 'wb'))	
	logging.info("training complete and model is saved")

def loadModel(learning_rate_val, max_depth_val):
	# Classifier Declared 
	xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = learning_rate_val, 
                         max_depth = max_depth_val, alpha = 5, n_estimators = 10)

def trainModel(learning_rate_val, max_depth_val, base_folder, clean):

	chunkcount = 1
	cleanDframe = True if clean==1 else False
	logging.info("Base folder:: clean dataframe "+base_folder+"::"+str(cleanDframe))
	df1, df2, df3, df4 = loadDatasets(cleanDframe)
	
	xg_reg = loadModel(learning_rate_val, max_depth_val)

	#Load the categorical columns for faster filling in between
	categoricalCols = [ 'merchant_id', 'slot_names', 'container_type', 'language_code',
							 'component_name', 'component_namespace', 'site']
	if not cleanDframe:
		categoricalCols.remove('language_code')
	categorySeries, categoryLists = {}, []
	for col in categoricalCols:
		tempSeries, tempList = loadCategorialList(base_folder, columnNm)
		categoryLists.append(tempList)
		categorySeries[col] = tempSeries

	one_hot_encoder = OneHotEncoder(categories=categoryLists, handle_unknown='ignore')	

	training_data_file = '/data/s3_file/3_10_files/full_metrics'
	for chunk in pd.read_csv(training_data_file, chunksize=CHUNKSIZE):
		logging.info("Start chunk Processing - " + str(chunkcount))
		chunkcount = chunkcount + 1 
		chunk.columns = ['frozen_placement_id', 'impressions', 'metrics_hour']

		#format the timestamp columns
		chunk['metrics_hour'] = pd.to_datetime(chunk['metrics_hour'], format='%Y %m %d %H:%M:%S')
		chunk['metrics_hour'] = chunk['metrics_hour'].dt.tz_localize(None)
		
		df4['start_date'] =  pd.to_datetime(df4['start_date'], format='%Y %m %d %H:%M:%S')
		df4['end_date'] =  pd.to_datetime(df4['end_date'], format='%Y %m %d %H:%M:%S')

		df_merged_set = mergeDataframe(chunk, df1, 'frozen_placement_id')
		df_merged_set = mergeDataframe(df_merged_set, df2, 'frozen_content_id')
		if not cleanDframe:
			# If dataframe is not cleaned then do not merge as it has only 1/2 M rows
			df_merged_set = mergeDataframe(df_merged_set, df3, 'frozen_content_id')
		
		df_merged_set = mergeDataframe(df_merged_set, df4, 'frozen_placement_id')
		if not cleanDframe:
			df_merged_set = df_merged_set.dropna()

		# Generate the days and hour interval time gaps
		deltaTime = (df_merged_set['metrics_hour'] - df_merged_set['start_date']).dt
		df_merged_set['days_interval']  = deltaTime.days
		df_merged_set['hours_interval'] = deltaTime.total_seconds()/3600
		df_merged_set['seconds_interval']  = deltaTime.total_seconds()

		columns_to_keep = ['impressions', 'merchant_id', 'slot_names', 
			'container_type', 'language_code', 'component_name', 'component_namespace', 'guarantee_percentage', 
			'site', 'container_id', 'days_interval', 'hours_interval', 'seconds_interval']
		# Remove language is data is not cleaned
		if not cleanDframe:
			columns_to_keep.remove('language_code')

		labelCols = ['container_id']
		numericCols = ['guarantee_percentage', 'days_interval', 'hours_interval', 'seconds_interval']

		# label container_id separately as too high cardinality
		df_merged_set = fillLabelFromFile(df_merged_set, labelCols[0])
		df_merged_set[labelCols[0]] = df_merged_set[labelCols[0]]/df_merged_set[labelCols[0]].max()
		
		for col in categoricalCols:
			df_merged_set = fillLabelFromFile(df_merged_set, col, categorySeries[col])

		X, Y = df_merged_set.iloc[:,1:], df_merged_set.iloc[:,0]
		one_hot_encoded = one_hot_encoder.transform(X[categoricalCols])
		# Drop the ctegorical columns 
		X = pd.concat([X[labelCols + numericCols], one_hot_encoded], axis=1)
		
		xg_reg.fit(X, Y)
	saveModel(xg_reg, label_encoder, learning_rate_val, max_depth_val)

def __main__():
	# count the arguments
	if len(sys.argv) < 5:
		raise RuntimeError("Please provode the learning_rate, max_depth, base folder and cleanDataframe(0/1)")
	trainModel(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

#This is required to call the main function
if __name__ == "__main__":
	__main__()