import sys
import boto3
import pickle
import logging
import datetime
import sagemaker

from os import path

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def updload(s3, bucket, prefix, filepath, filename):
	s3.Bucket(bucket).upload_file(filepath,  prefix + filename)


def download(s3, bucket, prefix, filepath):
	logging.info("Downloading :: bucket:prefix  "+bucket+":"+prefix)
	if path.exists(filepath):
		logging.info(filepath + " already exists")
		return filepath

	s3.Bucket(bucket).download_file(prefix, filepath)
	return filepath

def __main__():
	# count the arguments
	if len(sys.argv) < 4:
		raise RuntimeError("Please provide the function(upload/download), bucket, prefix, basefolder and file")
	logging.info("Params " + str(sys.argv))	

	logging.info('Program will auto exit after upload is completed')
	s3 = boto3.resource('s3')
	if(sys.argv[1] == 'upload'):
		updload(s3, sys.argv[2], sys.argv[3], sys.argv[4] + sys.argv[5], sys.argv[5])
	else:
		download(s3, sys.argv[2], sys.argv[3], sys.argv[4] + sys.argv[5])


#This is required to call the main function
if __name__ == "__main__":
	__main__()
