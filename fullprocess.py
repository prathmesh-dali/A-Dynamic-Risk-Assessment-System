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
import pandas as pd
from sklearn.metrics import f1_score
import training
import ingestion
import scoring
import deployment
import diagnostics
import reporting


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']

# Check and read new data
# first, read ingestedfiles.txt
ingestedfiles = []
with open(os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt')) as file:
    for line in file:
        ingestedfiles.extend(line.split(":")[-1].strip().split(" "))


# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
files = os.listdir(os.path.join(os.getcwd(), input_folder_path))
NEW_DATA = False
for file in files:
    if file not in ingestedfiles:
        NEW_DATA = True


# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if not NEW_DATA:
    print("No new data found")
    sys.exit()

ingestion.merge_multiple_dataframe()
# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data
OLD_F1SCORE = 0
with open(os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt')) as file:
    for line in file:
        OLD_F1SCORE = float(line.split('=')[-1].strip())

data = pd.read_csv(
    os.path.join(
        os.getcwd(),
        output_folder_path,
        'finaldata.csv'))
y = data.pop('exited')
X = data.drop(['corporation'], axis=1)

y_pred = diagnostics.model_predictions(X)

new_f1score = f1_score(y, y_pred)

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the
# process here

if new_f1score >= OLD_F1SCORE:
    print("Drift has not occured")
    sys.exit()

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

training.train_model()
scoring.score_model()
deployment.store_model_into_pickle()

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model

reporting.score_model()
subprocess.run(['python', 'apicalls.py'])
