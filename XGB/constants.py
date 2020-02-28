# Batch size of 10000
CHUNKSIZE = 250000

# Default value ACTUAL SIZE 75%
TRAIN_ITERATION = 30
TRAIN_MOD_COUNT = 10

CONSTANT_FILLER = 'missing'
NUMERIC_FILLER = 0
# All Customer filler
ALL_CONSUMER = 'allCustomer'

## Data Input Keys
IBUCKET_KEY = 'input_bucket'
IRESULT_PREFIX_KEY = 'result_prefix'

IFILES_KEY = 'files'
IFOLDER_KEY = 'base_folder'
ITRAINING_FP = 'training_file_name'
INUMERICAL_COLS = 'numrical_cols'
ICATEGORICAL_COLS = 'categorical_cols'
IRESULT_COL_KEY = 'result_col'
IRESULT_FUNCTION = 'result_function'
ISTARTEGY_KEY = 'startegy'

IPARAMS_KEY = 'learning_params'
ITRAIN_ITERATIONS = 'iterations'
ICHUNKSIZE_KEY = 'chunksize'
IOBJECTIVE_KEY = 'objective_key'

IMODEL_FP = "model_filepath"
IMODEL_FN = "model_filename"
IMULTIPLE_METRICS_FILES = 'multiple_metric_files'
IMULTIPLE_TRAINING_FILES = 'multiple_training_files'

#	Example data imput
#	data_input = {}
#	data_input[IBUCKET_KEY] = 'gautam.placement-metrics-prod'
#	data_input[IFILES_KEY] = [  
#			'deep-learning-ds/january/18Jan25JanPMetrics000',
#			'deep-learning-ds/january/18Jan25JanPMetadata000',
#			'deep-learning-ds/january/18Jan25JanCM000',
#			'deep-learning-ds/january/18Jan25JanPP000',
#			'deep-learning-ds/january/18Jan25JanCreative000'
#	]
# 	data_input[INUMERICAL_COLS] = []
#   data_input[ICATEGORICAL_COLS] = []
#   
#	data_input[IFOLDER_KEY] = base_folder
#	data_input[ITRAINING_FP] = base_folder + '/training_file'
