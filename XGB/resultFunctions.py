def use10Classifer(row):
	if(row['impressions']<10):
		return 0
	return 1

def useDigitClassifier5ValueSet(row):
	if(row['impressions']<10):
		return 0
	elif(row['impressions']<100):
		return 1
	elif(row['impressions']<1000):
		return 2
	elif(row['impressions']<10000):
		return 3
	return 4

def useDigitClassifier6ValueSet(row):
	if(row['impressions']<10):
		return 0
	elif(row['impressions']<100):
		return 1
	elif(row['impressions']<1000):
		return 2
	elif(row['impressions']<10000):
		return 3
	elif(row['impressions']<100000):
		return 4
	return 5	

def useDigitClassifier7ValueSet(row):
	if(row['impressions']<10):
		return 0
	elif(row['impressions']<100):
		return 1
	elif(row['impressions']<1000):
		return 2
	elif(row['impressions']<10000):
		return 3
	elif(row['impressions']<100000):
		return 4
	elif(row['impressions']<1000000):
		return 5	
	return 6
