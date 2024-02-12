"""
Author: Prathmesh Dali
Date: February, 2023
This file contains script to call apis and save the reposne in file
"""

import json
import os
import requests

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"


# Call each API endpoint and store the responses
response1 = requests.post(
    f"{URL}prediction", json={
        "dataset_path": "testdata/testdata.csv"}, timeout=100).text
response2 = requests.get(f"{URL}scoring", timeout=100).text
response3 = requests.get(f"{URL}summarystats", timeout=100).text
response4 = requests.get(f"{URL}diagnostics", timeout=100).text

# combine all API responses
responses = "Predictions: " + response1 + "\nF1 Score: " + response2 + "\nData Summary: " + response3 + "\n" + response4

# write the responses to workspace

with open('config.json', 'r') as f:
    config = json.load(f)

output_path = os.path.join(config['output_model_path'])

with open(os.path.join(os.getcwd(), output_path, "apireturns.txt"), "w") as returns_file:
    returns_file.write(responses)
