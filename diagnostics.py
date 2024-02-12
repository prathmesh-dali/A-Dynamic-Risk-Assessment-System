"""
Author: Prathmesh Dali
Date: February, 2023
This script calculates diagnostics summary of the model
"""

import timeit
import os
import json
import pickle
import subprocess
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

# Function to get model predictions


def model_predictions(X_df):
    '''
    This function loads saved model and returns the prediction based on the provided input
    '''
    logging.info("Making predictions for input data")
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    return model.predict(X_df)

# Function to get summary statistics


def dataframe_summary():
    '''
    This function calculates the summary of the ingested data such as mean, median, standard
    deviation for each column.
    '''
    logging.info("Computing data summary")
    df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            dataset_csv_path,
            'finaldata.csv'))
    non_numeric_columns = df.select_dtypes(include='object').columns.tolist()
    df.pop('exited')
    df = df.drop(columns=non_numeric_columns, axis=1)
    results = []
    for col in df.columns:
        results.append([col, "mean", df[col].mean()])
        results.append([col, "median", df[col].median()])
        results.append([col, "standard deviation", df[col].std()])
    return results


def missing_data():
    '''
    This function returns the percentage of NA cells in each column for ingested data
    '''
    logging.info("Computing NA percentage")
    df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            dataset_csv_path,
            'finaldata.csv'))
    nas = list(df.isna().sum())
    results = [nas[i] / len(df.index) for i in range(len(nas))]
    return results

# Function to get timings


def execution_time():
    '''
    This function return the execution time for the ingestion and training scripts
    '''
    logging.info("Computing execution time")
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - starttime

    starttime = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - starttime
    return [ingestion_time, training_time]

# Function to check dependencies


def outdated_packages_list():
    '''
    This function returns list of outdated packages along with the current version
    and latest version
    '''
    logging.info("Getting outdated modules")

    with open(os.path.join(os.getcwd(), 'requirements.txt'), 'r') as file:
        requirements = file.read().split('\n')
    requirements = [r.split('==')[0] for r in requirements if r]
    df = pd.DataFrame(requirements, columns=['Package'])
    df['Version'] = np.nan
    df['Latest'] = np.nan
    outdated_packages = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    outdated_packages = [row.split()[:-1] for row in outdated_packages.split('\n')[2:]]
    for pkg in outdated_packages:
        if(len(pkg)>0):
            df.loc[df['Package'] == pkg[0], ['Version','Latest']] = pkg[1:]
    df=df.dropna().reset_index(drop=True)
    return df.to_string(index=False)


if __name__ == '__main__':
    data_df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            test_data_path,
            'testdata.csv'))
    data_df = data_df.drop(['corporation', 'exited'], axis=1)
    model_predictions(data_df)
    dataframe_summary()
    execution_time()
    missing_data()
    outdated_packages_list()
