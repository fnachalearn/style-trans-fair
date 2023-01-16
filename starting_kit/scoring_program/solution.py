import os
from os.path import isfile
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split


# ================ Small auxiliary functions =================

def read_solutions(data_dir):
    ''' Function to read the Labels from CSV files'''


    #----------------------------------------------------------------
    # Settings
    #----------------------------------------------------------------
    TRAIN_CSV_PATH = os.path.join(data_dir,"train_labels.csv")
    TEST_CSV_PATH = os.path.join(data_dir,"test_labels.csv")

    #Check CSV file
    if not os.path.isfile(TRAIN_CSV_PATH):
        print('[-] CSV file Not Found')
        print('Make sure your dataset is in this format: https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat')
        return 

    if not os.path.isfile(TEST_CSV_PATH):
        print('[-] CSV file Not Found')
        print('Make sure your dataset is in this format: https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat')
        return 
    
    #----------------------------------------------------------------
    # Load CSV
    #----------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
        
    
    

    #----------------------------------------------------------------
    # Check Columns in CSV
    #----------------------------------------------------------------
    csv_columns = train_df.columns

    #FILE_NAME 
    if not 'FILE_NAME' in csv_columns:
        print('[FILE_NAME Column Not Found')
        return


    #CATEGORY 
    if not 'CATEGORY' in csv_columns:
        print('[CATEGORY Column Not Found')
        return

    #STYLE 
    if not 'STYLE' in csv_columns:
        print('[STYLE Column Not Found')
        return


    print("-------------------------------------")
    print("[+] Your dataset is in perfect format")
    print("-------------------------------------\n\n")
   

    print("###-------------------------------------###")
    print("### Loading Data")
    print("###-------------------------------------###\n\n")


    #----------------------------------------------------------------
    # Categories
    #----------------------------------------------------------------

    categories = train_df['CATEGORY'].unique()
    total_categories = len(categories)

    #----------------------------------------------------------------
    # Styles
    #----------------------------------------------------------------

    styles = train_df['STYLE'].unique()
    total_styles = len(styles)

    print("###-------------------------------------###")
    print("### Train solutions : ", train_df.shape[0])
    print("### Test solutions : ", test_df.shape[0])
    print("###-------------------------------------###\n\n")
    
    
    train_solution =  train_df['CATEGORY'].values
    test_solution = test_df['CATEGORY'].values

    train_styles =  train_df['STYLE'].values
    test_styles =  test_df['STYLE'].values
    
    solutions = [train_solution, test_solution]
    styles = [train_styles, test_styles]
    solution_names = ['train', 'test']
    
    print("###-------------------------------------###")
    print("### Solutions files are ready!")
    print("###-------------------------------------###\n\n")
    
    return (solution_names,solutions,styles)


