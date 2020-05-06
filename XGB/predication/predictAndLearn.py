import logging
import sys

sys.path.append('..')

# Log time-level and message for getting a running estimate
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def predict(data_input, training_filepath):

    # Predication will be always on 1 result col
    YColumns = [data_input[constants.IRESULT_COL_KEY]]
    numericalCols = data_input[constants.INUMERICAL_COLS]
    categoricalCols = data_input[constants.ICATEGORICAL_COLS]



    return 0
