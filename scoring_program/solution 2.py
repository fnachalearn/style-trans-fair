import os
from os.path import isfile
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import random


# ================ Small auxiliary functions =================

def read_solutions(data_dir, random_state=42):
    ''' Function to read the Labels from CSV files'''


    #----------------------------------------------------------------
    # Settings
    #----------------------------------------------------------------
    JSON_PATH = os.path.join(data_dir,"info.json")
    CSV_PATH = os.path.join(data_dir,"labels.csv")
    
    #Check JSON file
    if not os.path.isfile(JSON_PATH):
        print('[-] JSON file Not Found')
        print('Make sure your dataset is in this format: https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat')
        return

    #Check CSV file
    if not os.path.isfile(CSV_PATH):
        print('[-] CSV file Not Found')
        print('Make sure your dataset is in this format: https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat')
        return 
    

    #----------------------------------------------------------------
    # Read JSON
    #----------------------------------------------------------------
    f = open (JSON_PATH, "r")
    info = json.loads(f.read())


    #----------------------------------------------------------------
    # Load CSV
    #----------------------------------------------------------------
    data_df = pd.read_csv(CSV_PATH)
        
    
    

    #----------------------------------------------------------------
    # Check Columns in CSV
    #----------------------------------------------------------------
    csv_columns = data_df.columns





    #Category 
    if not info["category_column_name"] in csv_columns:
        print('[-] Column Not Found : ' + info["category_column_name"])
        return



    #Super Category 
    if info["has_super_categories"]:
        if not info["super_category_column_name"] in csv_columns:
            print('[-] Column Not Found : ' + info["super_category_column_name"])
            return


    #----------------------------------------------------------------
    # Settings from info JSON file
    #----------------------------------------------------------------

    # category column name in csv
    CATEGORY_COLUMN = info["category_column_name"]

    # style column name in csv
    STYLE_COLUMN = info["style_column_name"]

    #----------------------------------------------------------------
    # Load Labels
    #----------------------------------------------------------------

    data_df['label_cat'] = data_df[CATEGORY_COLUMN].astype('category')

    train_data, test_data = [], []
    for category in data_df[CATEGORY_COLUMN].unique():
        for style in data_df[STYLE_COLUMN].unique():
            train, test = train_test_split(data_df[(data_df[CATEGORY_COLUMN]==category)&(data_df[STYLE_COLUMN]==style)], test_size=0.5, random_state=random_state)
            train_data.append(train)
            test_data.append(test)
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)

    train_data = bias_spliter(train_data,percentage=60,random_state=random_state)

    print("###-------------------------------------###")
    print("### Train solutions : ", train_data.shape[0])
    print("### Test solutions : ", test_data.shape[0])
    print("###-------------------------------------###\n\n")
    
    train_solution =  np.asarray(train_data['label_cat'].cat.codes.values)
    test_solution = np.asarray(test_data['label_cat'].cat.codes.values)

    train_styles =  train_data[STYLE_COLUMN].values
    test_styles = test_data[STYLE_COLUMN].values
    
    solutions = [train_solution, test_solution]
    styles = [train_styles, test_styles]
    solution_names = ['train', 'test']
    
    print("###-------------------------------------###")
    print("### Solutions files are ready!")
    print("###-------------------------------------###\n\n")
    
    return solution_names, solutions, styles


    # #----------------------------------------------------------------
    # # Categories
    # #----------------------------------------------------------------

    # categories = train_df['CATEGORY'].unique()
    # total_categories = len(categories)

    # #----------------------------------------------------------------
    # # Styles
    # #----------------------------------------------------------------

    # styles = train_df['STYLE'].unique()
    # total_styles = len(styles)

    # print("###-------------------------------------###")
    # print("### Train solutions : ", train_df.shape[0])
    # print("### Test solutions : ", test_df.shape[0])
    # print("###-------------------------------------###\n\n")
    
    
    # train_solution =  train_df['CATEGORY'].values
    # test_solution = test_df['CATEGORY'].values

    # train_styles =  train_df['STYLE'].values
    # test_styles =  test_df['STYLE'].values
    
    # solutions = [train_solution, test_solution]
    # styles = [train_styles, test_styles]
    # solution_names = ['train', 'test']
    
    # print("###-------------------------------------###")
    # print("### Solutions files are ready!")
    # print("###-------------------------------------###\n\n")
    
    # return (solution_names,solutions,styles)


def bias_spliter(df, percentage = 60, random_state=42, shuffle_styles = True): 
    """
    The method is taking a dataframe and it is generating a biased set. 

    @df - the dataframe that needs to be biased 
    @percentage - the percentage of samples from the dominant style of each category 
    @shuffle_styles - if True, then the style will be randomly selected for each category

    return: a biased dataframe which has a dominant style for each category

    """
    categories = df['CATEGORY'].unique()
    styles = df['STYLE'].unique()
    
    if shuffle_styles : 
        random.Random(random_state).shuffle(styles)
        
    categories = categories.tolist() 
    styles = styles.tolist() 
    
    chunks = []
    
    for index_style, style in enumerate(styles):
        for index_category, category in enumerate(categories): 
            
            style_category_df = df[(df['STYLE'] == style) & (df['CATEGORY'] == category)]
            
            if index_style == index_category: 
                pivot = percentage*len(style_category_df)//100
                chunks.append(style_category_df[:pivot])
            else: 
                pivot = (100-percentage)//2*len(style_category_df)//100
                chunks.append(style_category_df[:pivot])
                                                
    
    biased_df = pd.concat(chunks, axis=0)                                            
        
    return biased_df
  