import sys
import boto3
import pickle
import logging
import sagemaker

import numpy as np
from os import path
import pandas as pd

import constants as const
import s3utils as s3utils

from enumclasses import MLFunction
from enumclasses import Startegy

def useChunk(mlFunction, startegy, chunkcount, maxTrainingCount=100):
	if(startegy== Startegy.Continous):
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

def removeNaN(df, categoricalCols, defValue):
	# Replace any NaN values
	for col in categoricalCols:
		df[[col]] = df[[col]].fillna(value=defValue)
	return df


# ----------------------------------------------------------------------------------------------------------
# def __main__():
# 	print("Testing train, Mod10, 4  "+str(useChunk(MLFunction.Train, Startegy.Mod10, 4)))
# 	print("Testing train, Mod10, 10 "+str(useChunk(MLFunction.Train, Startegy.Mod10, 10)))

# 	print("Testing not of  validate, Mod10, 34 "+str(not useChunk(MLFunction.Validate, Startegy.Mod10, 34)))
# 	print("Testing not of validate, Mod10, 20 "+str(not useChunk(MLFunction.Validate, Startegy.Mod10, 20)))


# if __name__ == "__main__":
# 	__main__()