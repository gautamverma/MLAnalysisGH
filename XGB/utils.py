import pickle
import pandas as pd

from enumclasses import MLFunction
from enumclasses import Startegy

from sklearn.preprocessing import OneHotEncoder

def useChunk(mlFunction, startegy, chunkcount, maxTrainingCount):
	if(startegy == Startegy.Continous):
		# MAX COUNT can't be null for the continous training
		if(maxTrainingCount is None):
			raise RuntimeError("Please provide max training count for iterations")

		if(mlFunction == MLFunction.Train):
			if(chunkcount<maxTrainingCount):
				return True
		elif(mlFunction ==  MLFunction.Validate):
			if(chunkcount>maxTrainingCount):
				return True
	elif(startegy == Startegy.Mod10):
		if(mlFunction == MLFunction.Train):
			return chunkcount%10 != 0
		elif(mlFunction == MLFunction.Validate):
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
# 	print("Testing train, Mod10, 4  "+str(useChunk(MLFunction.Train, Startegy.Mod10, 4)))
# 	print("Testing train, Mod10, 10 "+str(useChunk(MLFunction.Train, Startegy.Mod10, 10)))

# 	print("Testing not of  validate, Mod10, 34 "+str(not useChunk(MLFunction.Validate, Startegy.Mod10, 34)))
# 	print("Testing not of validate, Mod10, 20 "+str(not useChunk(MLFunction.Validate, Startegy.Mod10, 20)))


# if __name__ == "__main__":
# 	__main__()