import sys
import logging
from os import path

import boto3

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Prefix will also contain the filename which we build when calling it Path(prefix).stem
def downloadFileFromS3(s3, bucket, prefix, filepath):
	logging.info("Downloading :: bucket:prefix"+bucket+":"+prefix)
	if(path.exists(filepath)):
		logging.info(filepath + " already exists")
		return filepath

	s3.Bucket(bucket).download_file(prefix, filepath)
	return filepath

def uploadFiletoS3(bucket, prefix, filepath):
	s3 = boto3.resource('s3')
	if(path.exists(filepath)):
		logging.info("Uploading file : " + filepath)
		logging.info("Uploading :: bucket:prefix  " + bucket + ":" + prefix)
		s3.Bucket(bucket).upload_file(filepath, prefix)
		return filepath
	raise RuntimeError("file : " + filepath +" doesn't exists, so upload failed")