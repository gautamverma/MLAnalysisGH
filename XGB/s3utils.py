import logging
from os import path

import boto3


# Prefix will also contain the filename which we build when calling it Path(prefix).stem
def downloadFileFromS3(s3, bucket, prefix, filepath):
	if(path.exists(filepath)):
		logging.info(filepath + " already exists")
		return filepath

	s3.Bucket(bucket).download_file(prefix, filepath)
	return filepath

def uploadFiletoS3(bucket, prefix, filepath):
	s3 = boto3.resource('s3')
	if(path.exists(filepath)):
		logging.info("Uploading file : " + filepath)
		s3.Bucket('mybucket').upload_file(prefix, filepath)
		return filepath
	raise RuntimeError("file : " + filepath +" doesn't exists, so upload failed")