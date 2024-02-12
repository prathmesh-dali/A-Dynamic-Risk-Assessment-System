from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    data = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    y = data.pop('exited')
    X = data.drop(['corporation'], axis=1)

    with open(os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)

    f1score = metrics.f1_score(y, y_pred)

    with open(os.path.join(os.getcwd(),dataset_csv_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1score}")

    return f1score


if __name__ == '__main__':
    score_model()