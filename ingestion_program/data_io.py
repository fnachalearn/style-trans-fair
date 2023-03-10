# Functions performing various input/output operations for the ChaLearn AutoML challenge

# Main contributors: Arthur Pesah and Isabelle Guyon, August-October 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

import numpy as np
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
import json
import os
import shutil
from scipy.sparse import * # used in data_binary_sparse 
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import closing
from sys import stderr
from sys import version
from glob import glob as ls
from os import getcwd as pwd
from os.path import isfile
from pprint import pprint
import pkg_resources
import yaml
from shutil import copy2
import csv
import random
#import psutil
import platform

# ================ Small auxiliary functions =================

def read_data(data_dir, task_number, random_state=42):

    """ 
    Function to read the Images in raw format

    Argument:   data_dir 
                path to the data directory for example sample_data or public_data
    
    Return:     dict
                a dictionary with meta-data, images data and labels
    """



    print("###-------------------------------------###")
    print("### Checking Data")
    print("###-------------------------------------###\n\n")

    #----------------------------------------------------------------
    # Settings
    #----------------------------------------------------------------
    IMAGES_PATH = os.path.join(data_dir,"stylized")
    JSON_PATH = os.path.join(data_dir,"info.json")
    CSV_PATH = os.path.join(data_dir, "tasks", f"labels{task_number}.csv")


    #----------------------------------------------------------------
    # Errors
    #----------------------------------------------------------------

    # Check Images Directory
    if not os.path.exists(IMAGES_PATH):
        print('[-] Images directory Not Found')
        print('Make sure your dataset is in this format: https://github.com/fnachalearn/dataformat-style-trans-fair/blob/main/README.MD')
        return

    #Check JSON file
    if not os.path.isfile(JSON_PATH):
        print('[-] JSON file Not Found')
        print('Make sure your dataset is in this format: https://github.com/fnachalearn/dataformat-style-trans-fair/blob/main/README.MD')
        return

    #Check CSV file
    if not os.path.isfile(CSV_PATH):
        print('[-] CSV file Not Found')
        print('Make sure your dataset is in this format: https://github.com/fnachalearn/dataformat-style-trans-fair/blob/main/README.MD')
        return



    #----------------------------------------------------------------
    # Load CSV
    #----------------------------------------------------------------
    data_df = pd.read_csv(CSV_PATH)


    #----------------------------------------------------------------
    # Read JSON
    #----------------------------------------------------------------
    f = open (JSON_PATH, "r")
    info = json.loads(f.read())





    #----------------------------------------------------------------
    # Check Columns in CSV
    #----------------------------------------------------------------
    csv_columns = data_df.columns

    #Image 
    if not info["image_column_name"] in csv_columns:
        print('[-] Column Not Found : ' + info["image_column_name"])
        return


    #Category 
    if not info["category_column_name"] in csv_columns:
        print('[-] Column Not Found : ' + info["category_column_name"])
        return



    #Super Category 
    if info["has_super_categories"]:
        if not info["super_category_column_name"] in csv_columns:
            print('[-] Column Not Found : ' + info["super_category_column_name"])
            return


    print("-------------------------------------")
    print("[+] Your dataset is in perfect format")
    print("-------------------------------------\n\n")
   



    #----------------------------------------------------------------
    # Settings from info JSON file
    #----------------------------------------------------------------

    # category column name in csv
    CATEGORY_COLUMN = info["category_column_name"]

    # style column name in csv
    STYLE_COLUMN = info["style_column_name"]

    # image column name in csv
    IMAGE_COLUMN = info["image_column_name"]




    print("###-------------------------------------###")
    print("### Loading Data")
    print("###-------------------------------------###\n\n")
    

    

    #----------------------------------------------------------------
    # Categories
    #----------------------------------------------------------------

    categories = data_df[CATEGORY_COLUMN].unique()
    styles = data_df[STYLE_COLUMN].unique()
    total_categories = len(categories)
    total_styles = len(styles)

    #----------------------------------------------------------------
    # Load Images
    #----------------------------------------------------------------
    data_dict = {}
    
    data_df['label_cat'] = data_df[CATEGORY_COLUMN].astype('category')
    data_dict['categories'] = data_df['label_cat'].cat.categories.values

    data_dict['styles'] = data_df[STYLE_COLUMN].values

    data_dict['images'] = data_df[CATEGORY_COLUMN].value_counts().values


    train_data, test_data = [], []
    for category in data_df[CATEGORY_COLUMN].unique():
        for style in data_df[STYLE_COLUMN].unique():
            train, test = train_test_split(data_df[(data_df[CATEGORY_COLUMN]==category)&(data_df[STYLE_COLUMN]==style)], test_size=0.5, random_state=random_state)
            train_data.append(train)
            test_data.append(test)
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)

    train_data = bias_spliter(train_data,dominant_minority_ratio = 4,random_state=random_state)

    
    data_dict['train_labels'] = train_data[CATEGORY_COLUMN].values
    data_dict['test_labels'] = test_data[CATEGORY_COLUMN].values

    data_dict['train_styles'] = train_data[STYLE_COLUMN].values
    data_dict['test_styles'] = test_data[STYLE_COLUMN].values
    
    data_dict['train_labels_num'] =  np.asarray(train_data['label_cat'].cat.codes.values)
    data_dict['test_labels_num'] = np.asarray(test_data['label_cat'].cat.codes.values)
    
    
    data_dict['train_data'] = train_data[IMAGE_COLUMN].values
    data_dict['test_data'] = test_data[IMAGE_COLUMN].values

    data_dict['train_df'] = train_data
    data_dict['test_df'] = test_data

    print("-------------------------------------")
    print("[+] Data loaded successfully")
    print("-------------------------------------\n\n")

    print("###-------------------------------------###")
    print("### Loading Images")
    print("###-------------------------------------###\n\n")


    train_images = []
    test_images = []
    
    for image_name in data_dict['train_data']:
        file = IMAGES_PATH+"/"+image_name
        img = io.imread(file)
        train_images.append(img)

    for image_name in data_dict['test_data']:
        file = IMAGES_PATH+"/"+image_name
        img = io.imread(file)
        test_images.append(img)
        
    data_dict['train_images'] = np.asarray(train_images)
    data_dict['test_images'] = np.asarray(test_images)

    print("-------------------------------------")
    print("[+] Images loaded successfully")
    print("-------------------------------------\n\n")
    
    return data_dict, info

def read_as_df(basename, type="train"):
    ''' Function to read the AutoML format and return a Panda Data Frame '''
    csvfile = basename + '_' + type + '.csv'
    if isfile(csvfile):
        print('Reading '+ basename + '_' + type + ' from CSV')
        XY = pd.read_csv(csvfile)
        return XY
    	
    print('Reading '+ basename + '_' + type+ ' from AutoML format')
    feat_name = pd.read_csv(basename + '_feat.name', header=None)
    label_name = pd.read_csv(basename + '_label.name', header=None, names =['Class'])
    X = pd.read_csv(basename + '_' + type + '.data', sep=' ', names = np.ravel(feat_name))
    [patnum, featnum] = X.shape
    print('Number of examples = %d' % patnum)
    print('Number of features = %d' % featnum)
    
    XY=X
    Y=[]
    solution_file = basename + '_' + type + '.solution'
    if isfile(solution_file):
    	# This was reading the original multi-column 1-hot encoding
        Y = data(solution_file)
        [patnum2, classnum] = Y.shape
        assert(patnum==patnum2)
        if classnum==1:
            classnum=np.amax(Y)+1
            numerical_target=pd.DataFrame({'Class':Y[:,0].astype(int)})
        else:
            Y = pd.read_csv(solution_file, sep=' ', names = np.ravel(label_name))
            label_range = np.arange(classnum).transpose()         # This is just a column vector [[0], [1], [2]]
            numerical_target = Y.dot(label_range)                 # This is a column vector of dim patnum with numerical categories
        #print(numerical_target)
        # Here we add the target values as a last column, this is convenient to use seaborn
        # Look at http://seaborn.pydata.org/tutorial/axis_grids.html for other ideas
        #label_name = pd.DataFrame(['0', '1', '2'], columns=['col'])
        print(label_name)
        nominal_target = pd.Series(np.array(label_name)[numerical_target].ravel()) # Same with nominal categories
        print('Number of classes = %d' % classnum)
        XY = X.assign(target=nominal_target.values)          # Add the last column
    
    return XY
    
# ================ Small auxiliary functions =================

swrite = stderr.write

if (os.name == "nt"):
       filesep = '\\'
else:
       filesep = '/'
       


def bias_spliter(df, dominant_minority_ratio = 4, random_state=42, shuffle_styles = True): 
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
                chunks.append(style_category_df)
            else: 
                pivot = len(style_category_df) // dominant_minority_ratio
                chunks.append(style_category_df[:pivot])
                                                
    
    biased_df = pd.concat(chunks, axis=0)                                            
        
    return biased_df
  

def write_list(lst):
    ''' Write a list of items to stderr (for debug purposes)'''
    for item in lst:
        swrite(item + "\n") 
        
def print_dict(verbose, dct):
    ''' Write a dict to stderr (for debug purposes)'''
    if verbose:
        for item in dct:
            print(item + " = " + str(dct[item]))

def mkdir(d):
    ''' Create a new directory'''
    if not os.path.exists(d):
        os.makedirs(d)
        
def mvdir(source, dest):
    ''' Move a directory'''
    if os.path.exists(source):
        os.rename(source, dest)

def rmdir(d):
    ''' Remove an existingdirectory'''
    if os.path.exists(d):
        shutil.rmtree(d)
        
def vprint(mode, t):
    ''' Print to stdout, only if in verbose mode'''
    if(mode):
            print(t) 
        
# ================ Output prediction results and prepare code submission =================
        
def write(filename, predictions):
    ''' Write prediction scores in prescribed format'''
    with open(filename, "w") as output_file:
        for row in predictions:
                if type(row) is not np.ndarray and type(row) is not list:
                        row = [row]
                for val in row:
                        output_file.write('{0:g} '.format(float(val)))
                output_file.write('\n')

def zipdir(archivename, basedir, exclude_folders=[], exclude_files=[]):
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            if any([root.endswith(exclude) for exclude in exclude_folders]):
                continue
            #NOTE: ignore empty directories
            for fn in files:
                if any([fn.endswith(exclude) for exclude in exclude_files]):
                    continue
                if fn[-4:]!='.zip' and fn!='.DS_Store' :    
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(basedir):] #XXX: relative path
                    z.write(absfn, zfn)
                    
# ================ Inventory input data and create data structure =================
   
def inventory_data(input_dir):
    ''' Inventory the datasets in the input directory and return them in alphabetical order'''
    # Assume first that there is a hierarchy dataname/dataname_train.data
    training_names = inventory_data_dir(input_dir)
    ntr=len(training_names)
    if ntr==0:
        # Try to see if there is a flat directory structure
        training_names = inventory_data_nodir(input_dir)
    ntr=len(training_names)
    if ntr==0:
        print('WARNING: Inventory data - No data file found')
        training_names = []
    training_names.sort()
    return training_names
        
def inventory_data_nodir(input_dir):
    ''' Inventory data, assuming flat directory structure'''
    training_names = ls(os.path.join(input_dir, '*_train.data'))
    for i in range(0,len(training_names)):
        name = training_names[i]
        training_names[i] = name[-name[::-1].index(filesep):-name[::-1].index('_')-1]
        check_dataset(input_dir, training_names[i])
    return training_names
    
def inventory_data_dir(input_dir):
    ''' Inventory data, assuming flat directory structure, assuming a directory hierarchy'''
    training_names = ls(input_dir + '/*/*_train1.data') # This supports subdirectory structures obtained by concatenating bundles
    for i in range(0,len(training_names)):
        name = training_names[i]
        training_names[i] = name[-name[::-1].index(filesep):-name[::-1].index('_')-1]
        #check_dataset(os.path.join(input_dir, training_names[i]), training_names[i])
    return training_names
    
def check_dataset(dirname, name):
    ''' Check the test and valid files are in the directory, as well as the solution'''
    valid_file = os.path.join(dirname, name + '_valid.data')
    if not os.path.isfile(valid_file):
                print('No validation file for ' + name)
                exit(1)  
    test_file = os.path.join(dirname, name + '_test.data')
    if not os.path.isfile(test_file):
                print('No test file for ' + name)
                exit(1)
	# Check the training labels are there
    training_solution = os.path.join(dirname, name + '_train.solution')
    if not os.path.isfile(training_solution):
                print('No training labels for ' + name)
                exit(1)
    return True


def data(filename, nbr_features=None, verbose = False):
    ''' The 2nd parameter makes possible a using of the 3 functions of data reading (data, data_sparse, data_binary_sparse) without changing parameters'''
    if verbose: print (np.array(data_converter.file_to_array(filename)))
    return np.array(data_converter.file_to_array(filename), dtype=float)
            
def data_mv(filename, nbr_features=None, verbose = False):
    ''' The 2nd parameter makes possible a using of the 3 functions of data reading (data, data_sparse, data_binary_sparse) without changing parameters'''
    if verbose: print (np.array(data_converter.file_to_array_mv(filename)))
    return np.array(data_converter.file_to_array_mv(filename), dtype=float)

def data_sparse (filename, nbr_features):
    ''' This function takes as argument a file representing a sparse matrix
    sparse_matrix[i][j] = "a:b" means matrix[i][a] = basename and load it with the loadsvm load_svmlight_file
    '''
    return data_converter.file_to_libsvm (filename = filename, data_binary = False  , n_features = nbr_features)



def data_binary_sparse (filename , nbr_features):
    ''' This fuction takes as argument a file representing a sparse binary matrix 
    sparse_binary_matrix[i][j] = "a"and transforms it temporarily into file svmlibs format( <index2>:<value2>)
    to load it with the loadsvm load_svmlight_file
    '''
    return data_converter.file_to_libsvm (filename = filename, data_binary = True  , n_features = nbr_features)


 
# ================ Copy results from input to output ==========================
 
def copy_results(datanames, result_dir, output_dir, verbose):
    ''' This function copies all the [dataname.predict] results from result_dir to output_dir'''
    missing_files = []
    for basename in datanames:
        try:
            missing = False
            test_files = ls(result_dir + "/" + basename + "*_test*.predict")
            if len(test_files)==0: 
                vprint(verbose, "[-] Missing 'test' result files for " + basename) 
                missing = True
            valid_files = ls(result_dir + "/" + basename + "*_valid*.predict")
            if len(valid_files)==0: 
                vprint(verbose, "[-] Missing 'valid' result files for " + basename) 
                missing = True
            if missing == False:
                for f in test_files: copy2(f, output_dir)
                for f in valid_files: copy2(f, output_dir)
                vprint( verbose,  "[+] " + basename.capitalize() + " copied")
            else: 
                missing_files.append(basename)           
        except:
            vprint(verbose, "[-] Missing result files")
            return datanames
    return missing_files

# ================ Display directory structure and code version (for debug purposes) =================
      
def show_dir(run_dir):
	print('\n=== Listing run dir ===')
	write_list(ls(run_dir))
	write_list(ls(run_dir + '/*'))
	write_list(ls(run_dir + '/*/*'))
	write_list(ls(run_dir + '/*/*/*'))
	write_list(ls(run_dir + '/*/*/*/*'))
      
def show_io(input_dir, output_dir):     
	swrite('\n=== DIRECTORIES ===\n\n')
	# Show this directory
	swrite("-- Current directory " + pwd() + ":\n")
	write_list(ls('.'))
	write_list(ls('./*'))
	write_list(ls('./*/*'))
	swrite("\n")
	
	# List input and output directories
	swrite("-- Input directory " + input_dir + ":\n")
	write_list(ls(input_dir))
	write_list(ls(input_dir + '/*'))
	write_list(ls(input_dir + '/*/*'))
	write_list(ls(input_dir + '/*/*/*'))
	swrite("\n")
	swrite("-- Output directory  " + output_dir + ":\n")
	write_list(ls(output_dir))
	write_list(ls(output_dir + '/*'))
	swrite("\n")
        
    # write meta data to sdterr
	swrite('\n=== METADATA ===\n\n')
	swrite("-- Current directory " + pwd() + ":\n")
	try:
		metadata = yaml.load(open('metadata', 'r'))
		for key,value in metadata.items():
			swrite(key + ': ')
			swrite(str(value) + '\n')
	except:
		swrite("none\n");
	swrite("-- Input directory " + input_dir + ":\n")
	try:
		metadata = yaml.load(open(os.path.join(input_dir, 'metadata'), 'r'))
		for key,value in metadata.items():
			swrite(key + ': ')
			swrite(str(value) + '\n')
		swrite("\n")
	except:
		swrite("none\n");
	
def show_version():
	# Python version and library versions
	swrite('\n=== VERSIONS ===\n\n')
	# Python version
	swrite("Python version: " + version + "\n\n")
	# Give information on the version installed
	swrite("Versions of libraries installed:\n")
	pprint([d for d in pkg_resources.working_set], stream=stderr)
 
 # Compute the total memory size of an object in bytes

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

    # write the results in a csv file
def platform_score ( basename , mem_used ,n_estimators , time_spent , time_budget ):
# write the results and platform information in a csv file (performance.csv)
    with open('performance.csv', 'a') as fp:
        a = csv.writer(fp, delimiter=',')
        #['Data name','Nb estimators','System', 'Machine' , 'Platform' ,'memory used (Mb)' , 'number of CPU' ,' time spent (sec)' , 'time budget (sec)'],
        data = [
#        [basename,n_estimators,platform.system(), platform.machine(),platform.platform() , float("{0:.2f}".format(mem_used/1048576.0)) , str(psutil.cpu_count()) , float("{0:.2f}".format(time_spent)) ,    time_budget ]
        [basename,n_estimators,platform.system(), platform.machine(),platform.platform() , float("{0:.2f}".format(mem_used/1048576.0)) ,  float("{0:.2f}".format(time_spent)) ,    time_budget ]
        ]
        a.writerows(data)