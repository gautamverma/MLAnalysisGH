import sys
import pickle
import logging
import datetime

from os import path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

CONSTANT_FILLER = 'missing'
NUMERIC_FILLER = 0

def removeNaN(df, categoricalCols, defValue):
	# Replace any NaN values
	for col in categoricalCols:
		df[[col]] = df[[col]].fillna(value=defValue)
	return df


def exploreFile(filename):

	numericalCols = ['impressions', 'guarantee_percentage', 'container_id_label']
	categoricalCols = ["component_name", "slot_names", "container_type", "component_namespace","component_display_name", "customer_targeting", "site", "objective", "intent"]

	df = pd.read_csv(filename, skiprows=0, header=0, usecols=categoricalCols)
	logging.info("List of columns with null ")
	logging.info(df.columns[df.isnull().any()].tolist())

	logging.info("Shape "+str(df.shape))
	logging.info(str(df.weblab.value_counts()))

	columns_to_keep =  numericalCols + categoricalCols
	df = df[columns_to_keep + ['weblab']]

	logging.info(str(df.dtypes))	
	logging.info(str(df.container_id_label.value_counts()))	
	logging.info(str(df.guarantee_percentage.value_counts()))

	# Fill all Missing Values so dropna doesn't remove any row
	df = removeNaN(df, numericalCols, NUMERIC_FILLER)
	df = removeNaN(df, categoricalCols, CONSTANT_FILLER)

	logging.info("weblab missing data count : "+ str(df.where(df['weblab']=='missing').dropna().shape))

	filter10less = df['impressions']<10
	filter10more = df['impressions']>10

	filter100less = df['impressions']<100
	filter100more = df['impressions']>100

	filter1000less = df['impressions']<1000
	filter1000more = df['impressions']>1000

	filter10000less = df['impressions']<10000
	filter10000more = df['impressions']>10000

	filter100000less = df['impressions']<100000
	filter100000more = df['impressions']>100000

	logging.info("count < 10 impression : "+ str(df.where(filter10less).dropna().shape))
	logging.info("10 < count < 100 impression  : "+ str(df.where(filter10more & filter100less).dropna().shape))
	logging.info("100 < count < 1000 impression : "+ str(df.where(filter100more & filter1000less).dropna().shape))
	logging.info("1000 < count < 10000 impression : "+ str(df.where(filter1000more & filter10000less).dropna().shape))
	logging.info("10000 < count < 100000 impression : "+ str(df.where(filter100000more & filter100000less).dropna().shape))

def __main__():
	# count the arguments
	if len(sys.argv) < 2:
		raise RuntimeError("Please provide the filename")
	exploreFile(sys.argv[1])

#This is required to call the main function
if __name__ == "__main__":
	__main__()
