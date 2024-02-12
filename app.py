"""
Author: Prathmesh Dali
Date: February, 2023
This file contains apis for getting prediction and data summary
"""

import json
import os
from flask import Flask, request
import pandas as pd
from diagnostics import (
    model_predictions,
    dataframe_summary,
    execution_time,
    missing_data,
    outdated_packages_list)
from scoring import score_model


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    '''
    This endpoint return predictions for provided csv file
    '''
    filepath = request.json.get('dataset_path')
    df = pd.read_csv(os.path.join(os.getcwd(), filepath))
    X_df = df.drop(['corporation', 'exited'], axis=1)
    return str(model_predictions(X_df))

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    '''
    This endpoint return f1 score for test data
    '''
    return str(score_model())

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summary_stats():
    '''
    This endpoint retruns data summary such as mean, median, standard deviation for all columns
    '''
    return str(dataframe_summary())

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    '''
    This endpoint return the execution time, NA percentage and pip packages details
    '''
    et = execution_time()
    nas = missing_data()
    od = outdated_packages_list()
    return str(
        "execution_time: " +
        str(et) +
        "\nmissing_data: " +
        str(nas) +
        "\noutdated_packages: \n" +
        od)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
