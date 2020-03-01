import logging

import constants as const
import numpy as np
import pandas as pd
import xgboost as xgb
from enumclasses import MLFunction
from resultFunctions import callFunctionByName
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import utils as utils


def predictXGBModel(data_input, xg_reg):

    chunk_accuracy = {}
    YColumns = [data_input[const.IRESULT_COL_KEY]]
    numericalCols = data_input[const.INUMERICAL_COLS]
    categoricalCols = data_input[const.ICATEGORICAL_COLS]

    columns_to_keep = YColumns + numericalCols + categoricalCols
    one_hot_encoder, shapeTuple = utils.buildOneHotEncoder(data_input[const.ITRAINING_FP], categoricalCols)
    logging.info('One hot encoder is ready')

    chunkcount = 1
    total_chunk_count = shapeTuple[0] / data_input[const.ICHUNKSIZE_KEY]

    logging.info("Predicating for  " + data_input[const.IOBJECTIVE_KEY])
    logging.info("Predicating using stragegy : " + str(data_input[const.ISTARTEGY_KEY]))
    for chunk in pd.read_csv(data_input[const.ITRAINING_FP], chunksize=data_input[const.ICHUNKSIZE_KEY]):
        if not utils.useChunk(data_input[const.ISTARTEGY_KEY], MLFunction.Validate, chunkcount, total_chunk_count):
            chunkcount = chunkcount + 1
            continue

        logging.info('Starting Predication - ' + str(chunkcount))
        chunk[data_input[const.IRESULT_COL_KEY]] = chunk.apply(lambda row: callFunctionByName(row, data_input[const.IRESULT_FUNCTION]), axis=1)

        # Get only the columns to evaluate
        chunk = chunk[columns_to_keep + ['weblab']]

        # Get all rows where weblab is missing
        df_merged_set_test = chunk.where(chunk['weblab'] == "missing").dropna()
        df_merged_set_test = df_merged_set_test[columns_to_keep]
        logging.info('Weblab Removed: Shape - ' + str(df_merged_set_test.shape))

        INPUT = df_merged_set_test[numericalCols]
        ONEHOT = df_merged_set_test[categoricalCols]
        OUTPUT = df_merged_set_test[YColumns]

        one_hot_encoded = one_hot_encoder.transform(ONEHOT)
        logging.info('One hot encoding done for : '+str(chunkcount))

        dataMatrix = xgb.DMatrix(np.column_stack((INPUT, one_hot_encoded)))

        predictions = xg_reg.predict(dataMatrix)
        chunkcount = chunkcount + 1

        # Result Analysis for Chunk
        matrix = confusion_matrix(OUTPUT, np.around(predictions))
        accuracy = accuracy_score(OUTPUT, np.around(predictions))
        logging.info('Confusion Matrix : ' + str(matrix))
        logging.info('Accuracy Score : ' + str(accuracy))
        logging.info('Report : ')
        logging.info(str(classification_report(OUTPUT, np.around(predictions))))
        chunk_accuracy[chunkcount] = accuracy

    accuracy_fn = "model_accuracy_score.sav"
    return accuracy_fn, utils.saveDataOnDisk(accuracy, data_input[const.IFOLDER_KEY] + accuracy_fn)