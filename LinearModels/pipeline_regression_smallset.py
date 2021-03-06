import sys
import pickle
import logging
import datetime

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# This will only use the Label Encoder as on using the one hot encoding 
# we have to guarantee all the values in that one-hot-column has to present 
# in the test dataset also otherwise it will result in the feature mismatch bug

# TODO To resolve this we need put the columns in same order and intialize the old columns 
# We will do this in later in this we are training a simple XGB Model for the seven days data

CHUNKSIZE = 1000000

def mergeDataframe(df1, df2, column, joinType='inner'):
	if column is None:
		raise RuntimeError("Column can't be null. Please give the column value")
	return pd.merge(df1, df2, on=column, how=joinType);

# Here the cateogrical is constant for faster check
def imputeMissingCols(df, numericCols, categoricalCols):
	for col in numericCols:
		isNullPresent = pd.isnull(df.iloc[:,col]).any() 
		logging.info(isNullPresent)
		if isNullPresent:
			numericImputer = SimpleImputer(missing_values=np.nan, strategy='median')
			numericImputer.fit(df.iloc[:,col:col+1])	
			df.iloc[:,col:col+1] = numericImputer.transform(df.iloc[:,col:col+1])
			logging.info("Imputer complete for integer column")

	for col in categoricalCols:
		isNullPresent = pd.isnull(df.iloc[:,col]).any() 
		logging.info(isNullPresent)
		if isNullPresent:
			categoricalImputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='B1B1B1B1B1')
			categoricalImputer.fit(df.iloc[:,col:col+1])
			df.iloc[:,col:col+1] = categoricalImputer.transform(df.iloc[:,col:col+1])
			logging.info("Imputer complete for category column")
	return df;


def cleanDataframe(df):
	datatypes = list(df.dtypes.iteritems())
	categoricalCols, numericCols, iColNames, cColNames = [], [], [], []
	for  i in range(len(datatypes)):
		str_dtype = str(datatypes[i][1])
		if str_dtype.startswith('int') or str_dtype.startswith('float'):
			numericCols.append(i)
			iColNames.append(datatypes[i][0])
		else:
			categoricalCols.append(i)
			cColNames.append(datatypes[i][0])
	logging.info("Numeric Columns", str(iColNames))
	logging.info("Category Columns", str(cColNames))
	return imputeMissingCols(df, numericCols, categoricalCols);


def loadDatasets(cleanDF):
	logging.info("Start loading the Datasets")
	PlACEMENTS__FILENAME = '3_10_files/3_10NovemberDS_Placementfeature000'
	CONTENT_FILENAME = '3_10_files/3_10NovemberDS_Contentfeature000'
	RESOURCEBUNDLE_FILENAME = '3_10_files/3_10NovemberDS_Bundlefeature000'
	PLACEMENT_PROPERTIES_FILENAME = '3_10_files/3_10November0S_7feature000'

	# Make sure you download these file in the backed EC2 instance
	metrics_df1 = pd.read_csv('/data/s3_file/'+PlACEMENTS__FILENAME, skiprows=0, header=None)
	metrics_df1.columns = ['frozen_placement_id', 'frozen_content_id', 'guarantee_percentage', 'created_by']
	# for null guarantee fill the explicitely 0
	logging.info("guarantee null to explicit 0")
	if 'guarantee_percentage' in metrics_df1.columns:
		metrics_df1 = metrics_df1.replace(np.nan, 0)
	if cleanDF:
		logging.info("Cleaning dataframe df1")
		metrics_df1  = cleanDataframe(metrics_df1)

	metrics_df2 = pd.read_csv('/data/s3_file/'+CONTENT_FILENAME, skiprows=0, header=None)
	metrics_df2.columns = ['frozen_content_id', 'component_name', 'component_namespace', 'created_by']
	if cleanDF:
		logging.info("Cleaning dataframe df2")
		metrics_df2  = cleanDataframe(metrics_df2)

	metrics_df3 = pd.read_csv('/data/s3_file/'+RESOURCEBUNDLE_FILENAME, skiprows=0, header=None)
	metrics_df3.columns = ['frozen_content_id', 'language_code']
	if cleanDF:
		logging.info("Cleaning dataframe df3")
		metrics_df3  = cleanDataframe(metrics_df3)

	metrics_df4 = pd.read_csv('/data/s3_file/'+PLACEMENT_PROPERTIES_FILENAME, skiprows=0, header=None)
	metrics_df4.columns = ['frozen_placement_id', 'container_type', 'container_id', 'slot_names', 'merchant_id',
	 'site', 'weblab', 'bullseye', 'is_recognized', 'parent_browse_nodes', 'store_names','start_date', 'end_date']

	 # Decreased the file Size
	metrics_df4  = metrics_df4[['frozen_placement_id', 'container_type', 'container_id', 
				'slot_names', 'merchant_id', 'site','start_date', 'end_date']]
	
	if cleanDF:
		logging.info("Cleaning dataframe df4")
		metrics_df4  = cleanDataframe(metrics_df4)
	logging.info("Datafile loaded and cleaned")
	return metrics_df1, metrics_df2, metrics_df3, metrics_df4;

#This is used to encode the Y
def labelCategoryColumns(df, cols):
	label_encoder = LabelEncoder()

	for col in cols:
		df.loc[:,col] = label_encoder.fit_transform(df.loc[:,col]).astype('int64')
	return df;	


def saveModel(xg_reg, learning_rate_val, max_depth_val):
	filename =  '/data/models/linear_regression_small_set_02_01_2020_{}_{}_{}.sav'
	filename  = filename.format(learning_rate_val, max_depth_val, int(datetime.datetime.now().timestamp())) 
	pickle.dump(xg_reg, open(filename, 'wb'))
	logging.info("Model training complete and model is saved")

def trainModel():
	df1, df2, df3, df4 = loadDatasets(True)
	training_data_file = '/data/s3_file/3_10_files/full_metrics'
	for chunk in pd.read_csv(training_data_file, chunksize=CHUNKSIZE):
		logging.info("Starting chunk processing")
		chunk.columns = ['frozen_placement_id', 'impressions', 'metrics_hour']

		#format the timestamp columns
		chunk['metrics_hour'] = pd.to_datetime(chunk['metrics_hour'], format='%Y %m %d %H:%M:%S')
		chunk['metrics_hour'] = chunk['metrics_hour'].dt.tz_localize(None)
		
		df4['start_date'] =  pd.to_datetime(df4['start_date'], format='%Y %m %d %H:%M:%S')
		df4['end_date'] =  pd.to_datetime(df4['end_date'], format='%Y %m %d %H:%M:%S')

		df_merged_set = mergeDataframe(chunk, df1, 'frozen_placement_id')
		df_merged_set = mergeDataframe(df_merged_set, df2, 'frozen_content_id')
		df_merged_set = mergeDataframe(df_merged_set, df3, 'frozen_content_id')
		df_merged_set = mergeDataframe(df_merged_set, df4, 'frozen_placement_id')

		# Generate the days and hour interval time gaps
		deltaTime = (df_merged_set['metrics_hour'] - df_merged_set['start_date']).dt
		df_merged_set['days_interval']  = deltaTime.days
		df_merged_set['hours_interval'] = deltaTime.total_seconds()/3600
		df_merged_set['seconds_interval']  = deltaTime.total_seconds()

		columns_to_keep = ['impressions', 'created_by_x', 'merchant_id', 'slot_names', 'container_id',
			'container_type', 'language_code', 'component_name', 'component_namespace', 'guarantee_percentage', 
			'site', 'days_interval', 'hours_interval', 'seconds_interval']

		# We create the preprocessing pipelines for both numeric and categorical data
		#  created_by_x is not a useful information and its not the exhaustive set

		# merchant_id -- find the grouging and find a way to assign a way for new merchant_id
		# 
		categoricalCols = ['created_by_x', 'merchant_id', 'slot_names',
							'container_type', 'language_code', 'component_name', 'component_namespace',
							'site', 'container_id']

		numericCols = ['guarantee_percentage', 'days_interval', 'hours_interval', 'seconds_interval']

		numeric_transformer = Pipeline(steps=[
    		('scaler', StandardScaler())])


		categorical_transformer = Pipeline(steps=[
    		('onehot', OneHotEncoder(handle_unknown='ignore'))])

		preprocessor = ColumnTransformer(
    		transformers=[
        		('num', numeric_transformer, numericCols),
        		('cat', categorical_transformer, categoricalCols)])

		clf = Pipeline(steps=[('preprocessor', preprocessor),
						('classifier', LinearRegression())])

		df_merged_set = df_merged_set[columns_to_keep]
		X, Y = df_merged_set.iloc[:,1:], df_merged_set.iloc[:,0]
		clf.fit(X, Y)
	saveModel(clf, 0, 0)


def __main__():
	trainModel()

#This is required to call the main function
if __name__ == "__main__":
	__main__()	