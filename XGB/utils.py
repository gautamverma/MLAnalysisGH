import sys
import pickle
import logging
import pandas as pd

from enumclasses import MLFunction
from enumclasses import Startegy

from sklearn.preprocessing import OneHotEncoder

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def useChunk(mlFunction, startegy, chunkcount, maxTrainingCount):
	logging.info("MLFunc "+str(mlFunction))
	logging.info("Star " + str (startegy))
	if startegy == Startegy.Continous:
		# MAX COUNT can't be null for the continous training
		if(maxTrainingCount is None):
			raise RuntimeError('Please provide max training count for iterations')
		if(mlFunction == MLFunction.Train):
			if(chunkcount<maxTrainingCount):
				return True
		elif(mlFunction ==  MLFunction.Validate):
			if(chunkcount>maxTrainingCount):
				return True
	elif(startegy == Startegy.Mod10):
		if(mlFunction == MLFunction.Train):
			logging.info("Train "+str(chunkcount%10))
			return chunkcount%10 != 0
		elif(mlFunction == MLFunction.Validate):
			logging.info("Validate "+str(chunkcount%10))
			return chunkcount%10 == 0
	return False

def validateInput(keys, inputData):
	return True

def mergeDataframe(df1, df2, column, joinType='inner'):
	if column is None:
		raise RuntimeError("Column can't be null. Please give the column value")
	return pd.merge(df1, df2, on=column, how=joinType);

# Build the One hot encoder using all data
def buildOneHotEncoder(training_file_name, categoricalCols):
	one_hot_encoder = OneHotEncoder(sparse=False)
	df = pd.read_csv(training_file_name, skiprows=0, header=0, usecols=categoricalCols)

	one_hot_encoder.fit(df)
	return one_hot_encoder, df.shape

def removeNaN(df, categoricalCols, defValue):
	# Replace any NaN values
	for col in categoricalCols:
		df[[col]] = df[[col]].fillna(value=defValue)
	return df

def saveDataOnDisk(data, filepath):
	pickle.dump(data, open(filepath, 'wb'))
	return filepath

# ----------------------------------------------------------------------------------------------------------
# def __main__():
# 	#print("Testing train, Mod10, 4  "+str(useChunk(MLFunction.Train, Startegy.Mod10, 4, 200)))
# 	#print("Testing train, Mod10, 10 "+str(useChunk(MLFunction.Train, Startegy.Mod10, 10, 200)))
#
# 	print("Testing not of  validate, Mod10, 34 "+str(not useChunk(MLFunction.Validate, Startegy['Mod10'], 9, 200)))
# 	print("Testing not of validate, Mod10, 20 "+str(not useChunk(MLFunction.Validate, Startegy['Mod10'], 20, 200)))
#
#
# if __name__ == "__main__":
# 	__main__()