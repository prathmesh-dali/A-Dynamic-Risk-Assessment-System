import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    filenames = os.listdir(os.path.join(os.getcwd(),input_folder_path))
    df = pd.DataFrame()

    for filename in filenames:
        df_file = pd.read_csv(os.path.join(os.getcwd(),input_folder_path,filename))
        df = df.append(df_file).reset_index(drop=True)
    
    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'w') as file:
        file.write(f"Data Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} : {' '.join(filenames)}")

    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(os.path.join(os.getcwd(),output_folder_path,'finaldata.csv'), index=False)



if __name__ == '__main__':
    merge_multiple_dataframe()
