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

def saveFile(df, filenm, base_folder='/data/s3_file/'):
	filename = base_folder + "_{}_" + filenm
	column_filename = base_folder + "_{}_ColumnLabels" 
	filename  = filename.format(int(datetime.datetime.now().timestamp()))    

	columns_to_write = df.columns;

	pickle.dump(str(columns_to_write), open(column_filename, 'wb'))
	df_merged_set.to_csv(filename, columns=columns_to_write, index=False, header=False, mode='w')

"""
def loadDatasets(base_folder):
	logging.info("Starting the loading of the datasets")
	PlACEMENTS__FILENAME = base_folder + '3_10NovemberDS_Placementfeature000'
	CONTENT_FILENAME = base_folder + '3_10NovemberDS_Contentfeature000'
	RESOURCEBUNDLE_FILENAME = base_folder + '3_10NovemberDS_Bundlefeature000'
	PLACEMENT_PROPERTIES_FILENAME = base_folder + '3_10November0S_7feature000'

	# Make sure you download these file in the backed EC2 instance
	metrics_df1 = pd.read_csv(base_folder + PlACEMENTS__FILENAME, skiprows=0, header=None)
	metrics_df1.columns = ['frozen_placement_id', 'frozen_content_id', 'guarantee_percentage', 'created_by']
	# for null guarantee fill the explicitely 0
	if 'guarantee_percentage' in metrics_df1.columns:
		metrics_df1 = metrics_df1.replace(np.nan, 0)
	metrics_df1  = cleanDataframe(metrics_df1)
	saveFile(base_folder, PlACEMENTS__FILENAME)

	metrics_df2 = pd.read_csv(base_folder + CONTENT_FILENAME, skiprows=0, header=None)
	metrics_df2.columns = ['frozen_content_id', 'component_name', 'component_namespace', 'created_by']
	metrics_df2  = cleanDataframe(metrics_df2)
	saveFile(base_folder, CONTENT_FILENAME)

	metrics_df3 = pd.read_csv(base_folder + RESOURCEBUNDLE_FILENAME, skiprows=0, header=None)
	metrics_df3.columns = ['frozen_content_id', 'language_code']
	metrics_df3  = cleanDataframe(metrics_df3)
	saveFile(base_folder, RESOURCEBUNDLE_FILENAME)

	metrics_df4 = pd.read_csv(base_folder + PLACEMENT_PROPERTIES_FILENAME, skiprows=0, header=None)
	metrics_df4.columns = ['frozen_placement_id', 'container_type', 'container_id', 'slot_names', 'merchant_id', 'site',
	 'weblab', 'bullseye', 'is_recognized', 'parent_browse_nodes', 'store_names','start_date', 'end_date']

	# Removing the some misssing datasets to reduce running time
	metrics_df4  = metrics_df4[['frozen_placement_id', 'container_type', 'container_id', 
				'slot_names', 'merchant_id', 'site','start_date', 'end_date']]

	metrics_df4  = cleanDataframe(metrics_df4)
	saveFile(base_folder, PLACEMENT_PROPERTIES_FILENAME)
	logging.info("Dataset cleaned and loaded")
"""

def loadDatasets(base_folder, filename):
	df = pd.read_csv(base_folder + filename, skiprows=0, header=None)
	df.columns = ['frozen_placement_id', 'frozen_content_id', 'guarantee_percentage', 'created_by']
	metrics_df1  = cleanDataframe(metrics_df1)
	saveFile(base_folder, PlACEMENTS__FILENAME)

def __main__():
	# count the arguments
	if len(sys.argv) < 3:
		raise RuntimeError("Please provide the base_folder and filename")
	trainModel(sys.argv[1])

#This is required to call the main function
if __name__ == "__main__":
	__main__()


