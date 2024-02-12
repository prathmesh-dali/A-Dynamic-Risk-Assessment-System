"""
Author: Prathmesh Dali
Date: February, 2023
This script deploys model to deployment folder
"""

import os
import json
import shutil
import logging

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


# function for deployment
def store_model_into_pickle():
    '''
    This function copies latest pickle file, the latestscore.txt value, and the
    ingestfiles.txt file into the deployment directory
    '''

    logging.info("Deploying the resources")
    shutil.copy(os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl'),
                os.path.join(os.getcwd(), prod_deployment_path))

    shutil.copy(os.path.join(os.getcwd(), dataset_csv_path, 'latestscore.txt'),
                os.path.join(os.getcwd(), prod_deployment_path))

    shutil.copy(
        os.path.join(
            os.getcwd(),
            dataset_csv_path,
            'ingestedfiles.txt'),
        os.path.join(
            os.getcwd(),
            prod_deployment_path))
    logging.info("Deployed the resources")


if __name__ == '__main__':
    store_model_into_pickle()
