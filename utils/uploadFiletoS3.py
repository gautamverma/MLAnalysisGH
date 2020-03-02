import sys
import boto3
import pickle
import logging
import datetime
import sagemaker

from os import path
import pandas as pd
import numpy as np
from pathlib import Path


def saveFileS3(s3, bucket, prefix, filepath, model_filename):
	s3.Bucket(input_bucket).upload_file(filepath,  prefix + model_filename)


def __main__():
	# count the arguments
	if len(sys.argv) < 4:
		raise RuntimeError("Please provide the bucket, prefix, basefolder and file")
	logging.info("Params " + str(sys.argv))	

	logging.info('Program will auto exit after upload is completed')
	s3 = boto3.resource('s3')
	saveFileS3(s3, sys.argv[1], sys.argv[2], sys.argv[3] + sys.argv[4], sys.argv[4])

#This is required to call the main function
if __name__ == "__main__":
	__main__()
