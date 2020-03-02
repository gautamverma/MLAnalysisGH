import sys
import pickle
import logging

import numpy as np
import pandas as pd

from os import path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# CONSTANTS
CONSTANT_FILLER = 'unknown_flag'

def save(filename, file_content_dict):
	pickle.dump(file_content_dict, open(filename, 'wb'))

def load(filename):
	return pickle.load(open(filename, 'rb'))

def loadJSON(parameter_filepath):
	data = {}
	with open(parameter_filepath, "r") as read_file:
    	data = json.load(read_file)
    return data

# Load the Labels Vocabulary in One Hot Encoder
def loadCategorialSeries(base_folder, columnNm):
	map_file = base_folder + columnNm + "_map.dict"
	if not path.exists(map_file):
		raise RuntimeError("Map file missing for "+ columnNm + " name")

	column_dict_file = open(map_file, "rb")
	column_dict = pickle.load(column_dict_file)
	column_series = pd.Series(column_dict)
	return column_series, list(column_series.index)

def labelCategoricalColumn(df, columnNm, columnSeries):
	logging.info(columnSeries.head())

	label = LabelEncoder()
	label.fit(columnSeries.to_numpy())
	# Prepare a dictionary for it 
	label_dict = dict(zip(label.classes_, label.transform(label.classes_)))

	# -1 is given to unknown classes
	return df[columnNm].apply(lambda x: label_dict.get(x, -1))

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

