"""
Author: Prathmesh Dali
Date: February, 2023
This file contains script check if new data ingested if ingested 
then check for drift and redeploy the model
"""

import os
import json
import subprocess
import sys
import logging
import pandas as pd
from sklearn.metrics import f1_score
import training
import ingestion
import scoring
import deployment
import diagnostics
import reporting

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']

logging.info("Reading the already ingested files")
ingestedfiles = []
with open(os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt')) as file:
    for line in file:
        ingestedfiles.extend(line.split(":")[-1].strip().split(" "))


logging.info("Checking if new files are already ingested")
files = os.listdir(os.path.join(os.getcwd(), input_folder_path))
NEW_DATA = False
for file in files:
    if file not in ingestedfiles:
        NEW_DATA = True


if not NEW_DATA:
    logging.info("No new data found")
    sys.exit()

logging.info("Ingesting new data")
ingestion.merge_multiple_dataframe()

logging.info("Reading previous f1 score")
OLD_F1SCORE = 0
with open(os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt')) as file:
    for line in file:
        OLD_F1SCORE = float(line.split('=')[-1].strip())

logging.info("Reading newly ingested data")
data_df = pd.read_csv(
    os.path.join(
        os.getcwd(),
        output_folder_path,
        'finaldata.csv'))
y_df = data_df.pop('exited')
X_df = data_df.drop(['corporation'], axis=1)

logging.info("Getting predictions for newly ingested data")
y_pred = diagnostics.model_predictions(X_df)

logging.info("Computing f1 score for newly ingested data")
new_f1score = f1_score(y_df, y_pred)

logging.info("Old F1 score %s", OLD_F1SCORE)
logging.info("New F1 score %s", new_f1score)

if new_f1score < OLD_F1SCORE:
    logging.info("No drift observed in the data")
    sys.exit()

logging.info("Traing model on new data")
training.train_model()

logging.info("Computing scores for newly trained model")
scoring.score_model()

logging.info("Deploying the newly created model")
deployment.store_model_into_pickle()

logging.info("Generating report for newly created model")
reporting.generate_report()

logging.info("Recoring API response for newly created model")
subprocess.run(['python', 'apicalls.py'])
