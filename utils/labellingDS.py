import sys
import pickle
import logging
import datetime

from os import path
import pandas as pd
import numpy as np

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

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

def __main__():
	# count the arguments
	if len(sys.argv) < 3:
		raise RuntimeError("Please provode the base folder and column name")
	logging.info("Folder name :: column name " + sys.argv[1] +" :: " + sys.argv[2])	
	generateLabel(sys.argv[1], sys.argv[2])

#This is required to call the main function
if __name__ == "__main__":
	__main__()