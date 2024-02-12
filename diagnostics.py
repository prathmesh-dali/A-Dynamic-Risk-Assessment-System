
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)
    return model.predict(data)#return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    data = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv'))
    non_numeric_columns = data.select_dtypes(include='object').columns.tolist()
    data.pop('exited')
    data = data.drop(columns=non_numeric_columns, axis=1)
    results =[]
    for col in data.columns:
        results.append([col, "mean", data[col].mean()])
        results.append([col, "median", data[col].median()])
        results.append([col, "standard deviation", data[col].std()])
    return results#return value should be a list containing all summary statistics

def missing_data():
    data = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, 'finaldata.csv'))
    nas = list(data.isna().sum())
    results = [nas[i]/len(data.index) for i in range(len(nas))]
    return results

##################Function to get timings
def execution_time():
    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - starttime

    starttime = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - starttime
    #calculate timing of training.py and ingestion.py
    return [ingestion_time, training_time]#return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


if __name__ == '__main__':
    data = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    data = data.drop(['corporation', 'exited'], axis=1)
    model_predictions(data)
    print(dataframe_summary())
    execution_time()
    missing_data()
    outdated_packages_list()





    
