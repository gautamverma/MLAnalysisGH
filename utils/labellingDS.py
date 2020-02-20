import sys
import pickle
import logging
import datetime

from os import path
import pandas as pd
import numpy as np

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

CONSTANT_FILLER='missing'

def generateLabel(folder , columnNm):
	# Check if the map file already generated
	map_file = folder + columnNm + "_map.dict"
	if path.exists(map_file):
		return loadedMap

	data_file = folder + columnNm + "_unique.csv"
	df = pd.read_csv(data_file, skiprows=0, header=0)
	if df.columns[0] != columnNm:
		raise RuntimeError("Column values don't match. Please check once")
	print(df.head())

	df[columnNm] = pd.Categorical(df[columnNm])
	df[columnNm + '_label'] = df[columnNm].cat.codes	
	print(df.head())
	column_series = pd.Series(df[columnNm + '_label'].values, index=df[columnNm]) 
	pickle.dump(column_series.to_dict(), open(map_file, 'wb'))

def generateLabelfromDf(baseFolder, columnNm, filepath):

	map_file = baseFolder + columnNm + "_map.dict"
	if path.exists(map_file):
		logging.info("Label unique hash file exists for col: "+ columnNm)
		return

	df = pd.read_csv(baseFolder + filepath, skiprows=0, header=0)
	df[[columnNm]] = df[[columnNm]].fillna(value=CONSTANT_FILLER)

	unique_column_list = df[columnNm].unique().tolist()
	logging.info("unique list head "+str(unique_column_list[:5]))

	unique_column_hash = {}
	position = 1
	for val in unique_column_list:
		unique_column_hash[val] = position
		position = position + 1

	logging.info('Label done for '+ columnNm)
	pickle.dump(unique_column_hash, open(map_file, 'wb'))
	return


def __main__():
	# count the arguments
	if len(sys.argv) < 3:
		raise RuntimeError("Please provode the method name , base folder and column name")
	logging.info("Folder name :: column name " + sys.argv[1] +" :: " + sys.argv[2])	
	if(sys.argv[1]=='generateLabel'):
		generateLabel(sys.argv[2], sys.argv[3])
	else:
		generateLabelfromDf(sys.argv[2], sys.argv[3], sys.argv[4])

#This is required to call the main function
if __name__ == "__main__":
	__main__()