"""
Author: Prathmesh Dali
Date: February, 2023
This script plots reports such as confusion matrix
"""

import json
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from diagnostics import model_predictions


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])


# Function for reporting
def score_model():
    '''
    This function calculates the confusion matric based on test data and
    saves plot for the same
    '''
    data_df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            test_data_path,
            'testdata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)
    y_pred = model_predictions(X_df)
    confusion_matrix = metrics.confusion_matrix(y_df, y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_matrix, cmap=plt.cm.Oranges, alpha=0.2)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j,
                    y=i,
                    s=confusion_matrix[i,
                                       j],
                    va='center',
                    ha='center',
                    size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(
        os.path.join(
            os.getcwd(),
            dataset_csv_path,
            "confusionmatrix.png"))


if __name__ == '__main__':
    score_model()
