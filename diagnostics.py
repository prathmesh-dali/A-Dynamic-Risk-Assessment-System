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
import pandas as pd

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
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    return model.predict(X_df)

# Function to get summary statistics


def dataframe_summary():
    '''
    This function calculates the summary of the ingested data such as mean, median, standard
    deviation for each column.
    '''
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
    outdated_packages = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


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
