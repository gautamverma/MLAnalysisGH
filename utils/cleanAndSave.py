import sys
import pickle
import logging
import datetime

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

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
	filename = base_folder + "_clean_" + filenm
	logging.info("Saving file " + filename)
	df.to_csv(filename, columns=df.columns, index=False, header=True, mode='w')

def loadDataset(base_folder, filename, columns):
	df = pd.read_csv(base_folder + filename, skiprows=0, header=None)
	df.columns = columns
	df  = cleanDataframe(df)
	saveFile(df, filename, base_folder)

def __main__():
	# count the arguments
	if len(sys.argv) < 4:
		raise RuntimeError("Please provide the base_folder, filename and columns(comma separated without quotes)")

	columns = sys.argv[3].strip('][').split(',')
	loadDataset(sys.argv[1], sys.argv[2], columns)

#This is required to call the main function
if __name__ == "__main__":
	__main__()


