"""
Author: Prathmesh Dali
Date: February, 2023
This script used for training Linear regression model on ingested data
"""

import json
import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():
    '''
    This function uses ingested data to train the logistic regression model
    '''
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data

    data_df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            dataset_csv_path,
            'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    model.fit(X_df, y_df)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl

    pickle.dump(
        model,
        open(
            os.path.join(
                os.getcwd(),
                model_path,
                'trainedmodel.pkl'),
            'wb'))


if __name__ == '__main__':
    train_model()
