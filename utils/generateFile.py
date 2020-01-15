import sys
import pickle

import pandas as pd

def save(filename, file_content_dict):
	pickle.dump(file_content_dict, open(filename, 'wb'))

def __main__():
	# count the arguments
	if len(sys.argv) < 4:
		raise RuntimeError("Please provode the filename, type and value")
	
	filename = sys.argv[1]
	value = None
	# list 1,2,3,4. --> ['1',''2,'3','4']
	# json key1=value1,key2=value2 --> {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
	if (sys.argv[2]=='list'):
		value = sys.argv[3].split(',')
	elif(sys.argv[2]=='json'):
		value = {}
		values = sys.argv[3].split(',')
		for val in values:
			keyValue = val.split("=")
			value[''+keyValue[0]] = keyValue[1]
	else:
		value = sys.argv[3]
	save(filename, value)

#This is required to call the main function
if __name__ == "__main__":
	__main__()