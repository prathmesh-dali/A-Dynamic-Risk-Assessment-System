"""
Author: Prathmesh Dali
Date: February, 2023
This script used for calculating and storing metric of the model based on test data
"""
import pickle
import os
import json
import logging
import pandas as pd
from sklearn import metrics

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():
    '''
    This funcion calculates the f1 score for the model and stores it in file
    '''
    logging.info("Calculating f1 score")
    data_df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            test_data_path,
            'testdata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    with open(os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_df)

    f1score = metrics.f1_score(y_df, y_pred)

    with open(os.path.join(os.getcwd(), dataset_csv_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1score}")

    logging.info("F1 score: %s", f1score)
    return f1score


if __name__ == '__main__':
    score_model()
