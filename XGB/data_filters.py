import sys
import logging

import constants as const
import numpy as np
import pandas as pd

# Log time-level and message for getting a running estimate
logging.basicConfig (stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

READ_CHUNK_SIZE = 10000000

def updateModelNmAndFilePath(data_input, model_prefix):
    # Update the Model name and filepath
    data_input[const.IFILE_PREFIX] = model_prefix
    data_input[const.IMODEL_FN] = model_prefix + data_input[const.IMODEL_FN]
    data_input[const.IMODEL_FP] = data_input[const.IFOLDER_KEY] + data_input[const.IMODEL_FN]
    return data_input

def filterProdEnviroment(data_input, training_file):
    data_input[const.PROD_ENVIROMENT_FILTERED_FILE] = data_input[const.IFOLDER_KEY] + 'filterAutoCreatedRecord'

    chunkcount = 1
    for df in pd.read_csv (training_file, skiprows=0, header=0, chunksize=READ_CHUNK_SIZE):
        logging.info ("Chunk shape " + str (df.shape))

        # Filter the placements created by the Prod enviroment for detail page
        filter = df['display_name'].str.startswith ('A+')

        df = df.mask (filter).dropna ()
        logging.info ("Prod data Filtered file shape " + str (df.shape))
        # It recreates the file if it is present
        if(chunkcount==1):
            df.to_csv(data_input[const.PROD_ENVIROMENT_FILTERED_FILE], index=False, encoding='utf-8')
        else:
            df.to_csv(data_input[const.PROD_ENVIROMENT_FILTERED_FILE], header=False, index=False, encoding='utf-8', mode='a')
        chunkcount = chunkcount + 1

    logging.info('Prod data filtered file created')
    data_input = updateModelNmAndFilePath(data_input, "ProdFilteredModel_")
    return data_input


def filterNonMarketingData(data_input, training_file):
    data_input[const.NON_MARKETING_FILTERED_FILE] = data_input[const.IFOLDER_KEY] + 'filterNonMarketingContent'

    non_marketing_component_names = ['AdPlacementsBlackjackATFWidget', 'AdPlacementsDPXWidget', 'AdPlacementsWidget',
                                     'AdSlotWidget', 'AdTechPlacementsBlackjackWidget', 'AdTechPlacementsWidget',
                                     'AndonCordPulling', 'AplusProductDescription', 'Aplus3pProductDescription',
                                     'AudibleCustomerNotificationsWidget', 'AudibleSEOPageSlot',
                                     'AudibleSocialMetaTagsWidget','AutoRotateHerotator', 'AutoRotateHerotatorV2', 'AutoRotateHerotatorV3',
                                     'DpxBlacklist', 'Dummy', 'DynamicWidget', 'EnumclawSlotDisplayAd',
                                     'ExtendedSearchCard','FetchBlackjackCategoryAssets', 'GiveawayMysteryCard', 'HeroOrder',
                                     'IssuanceBannerWidget', 'JarnesTest', 'KcpAppsGetForPlatform', 'MapleWidget',
                                     'PersonalizedContent','PrefetchResources', 'PrimenowAlertMessage', 'QtipData', 'Remote', 'RpBIA',
                                     'SeoTitleMeta', 'SimpleSnowAnnouncement', 'TimelineCard',
                                     'TypDesktopThankYouRecommendations','WeblabValidation', 'ZergnetWidget', 'audibleCSMMarkerWidget','audibleWebProductSummaries']
    chunkcount = 1
    for df in pd.read_csv (training_file, skiprows=0, header=0, chunksize=READ_CHUNK_SIZE):
        logging.info ("Full file shape " + str (df.shape))

        df = df[~df.component_name.isin(non_marketing_component_names)]
        logging.info ("Filter non markleting chunk shape " + str (df.shape))

        # It recreates the file if it is present
        if (chunkcount == 1):
            df.to_csv (data_input[const.NON_MARKETING_FILTERED_FILE], index=False, encoding='utf-8')
        else:
            df.to_csv (data_input[const.NON_MARKETING_FILTERED_FILE], header=False, index=False, encoding='utf-8',
                       mode='a')
        chunkcount = chunkcount + 1

    logging.info("Non Marketing full file created")
    data_input = updateModelNmAndFilePath(data_input, "NonMarketingFilteredModel_")
    return data_input;

# ----------------------------------------------------------------------------------------------------------
def __main__():

    data_input = {
        'base_folder': '/Users/gautamve/workspace/tempcsv/',
        'training_file_name': '/Users/gautamve/workspace/tempcsv/Book1.csv'
    }

    print("Prod filter data "+str(filterProdEnviroment(data_input)))
    print("Non marketing filter data "+str(filterNonMarketingData(data_input)))


if __name__ == "__main__":
	__main__()