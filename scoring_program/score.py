#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os
import glob
import json
import time
from sys import argv
from sklearn.metrics import accuracy_score

import libscores
import my_metric
import yaml
from libscores import *
import solution       
from solution import read_solutions
from sklearn import metrics
import matplotlib.pyplot as plt
import base64

# Default I/O directories:
root_dir = "../"
default_solution_dir = root_dir + "input_data"
default_prediction_dir = root_dir + "sample_result_submission"
default_score_dir = root_dir + "scoring_output"
default_data_name = "style_trans_fair_challenge"

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 0

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0

# =============================== MAIN ========================================

if __name__ == "__main__":
    start = time.time()

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default data directories if no arguments are provided
        solution_dir = default_solution_dir
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
        data_name = default_data_name
    elif len(argv) == 3: # The current default configuration of Codalab
        solution_dir = os.path.join(argv[1], 'ref')
        prediction_dir = os.path.join(argv[1], 'res')
        score_dir = argv[2]
        data_name = default_data_name
    elif len(argv) == 4:
        solution_dir = argv[1]
        prediction_dir = argv[2]
        score_dir = argv[3]
        data_name = default_data_name
    else: 
        swrite('\n*** WRONG NUMBER OF ARGUMENTS ***\n\n')
        exit(1)

    import solution       
    from solution import read_solutions
        
    # Create the output directory, if it does not already exist and open output files
    mkdir(score_dir)

    # Get the metric
    metric_name, scoring_function = get_metric()
    print("###-------------------------------------###")
    print("### Using metric : ", metric_name)
    print("###-------------------------------------###\n\n")
    
    
    #Solution Arrays
    # 3 arrays: train, validation and test
    for task_n in range(10):
        # Check if a file exists
        if not os.path.isfile(os.path.join(solution_dir, "tasks", f"labels{task_n}.csv")):
            continue
        try:
            score_file = open(os.path.join(score_dir, f'scores{task_n}.json'), 'w')
            score_json = {}
            html_file = open(os.path.join(score_dir, f'scores{task_n}.html'), 'w')


            solution_names, solutions, styles = read_solutions(solution_dir, task_number=task_n)

            html_file.write("<h3>Scoring output of your submission</h3>")
            for i, solution_name in enumerate(solution_names):
                
                set_num = i + 1  # 1-indexed
                score_name = 'set%s_score' % set_num
                try:
                    # Get the train prediction from the res subdirectory (must end with '.predict')
                    predict_file = os.path.join(prediction_dir, data_name + '_'+solution_name+f'{task_n}.predict')
                    if not os.path.isfile(predict_file):
                        print("#--ERROR--# "+solution_name.capitalize()+" predict file NOT Found!")
                        raise IOError("#--ERROR--# "+solution_name.capitalize()+" predict file NOT Found!")
                    # Read the solution and prediction values into numpy arrays
                    prediction = read_array(predict_file)
                    solution = solutions[i]
                    style = styles[i]
                    if (len(solution) != len(prediction)): 
                        print("#--ERROR--# Prediction length={} and Solution length={}".format(len(prediction), len(solution)))
                        raise ValueError("Prediction length={} and Solution length={}".format(len(prediction), len(solution)))
                    try:
                        # Compute the score prescribed by the metric file 
                        score = scoring_function(solution, prediction, style)
                        print(
                            "======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name + "): " + metric_name + "(" + score_name + ")=%0.12f =======" % score)
                        html_file.write(
                            "<pre>======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name + "): " + metric_name + "(" + score_name + ")=%0.12f =======\n</pre>" % score)
                        # Plot the confusion matrix, save it in a file, encode it, put it in the html and delete the file
                        conf_matrix=metrics.confusion_matrix(solution,prediction)
                        fig, ax = plt.subplots(figsize=(7.5, 7.5))
                        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
                        for i in range(conf_matrix.shape[0]):
                            for j in range(conf_matrix.shape[1]):
                                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
                        plt.xlabel('Predictions', fontsize=18)
                        plt.ylabel('Actuals', fontsize=18)
                        plt.title('Confusion Matrix', fontsize=18)
                        plt.savefig("confusion_matrix.png")
                        try:
                            filepath="confusion_matrix.png"
                            binary_fc = open(filepath, 'rb').read()
                            base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
                            ext = filepath.split('.')[-1]
                            dataurl = f'data:image/{ext};base64,{base64_utf8_str}'
                            html_file.write("Confusion Matrix:")
                            html_file.write("<img src="+dataurl+" alt='Confusion Matrix' width='250'/>")
                            html_file.write("<br>")
                            os.remove(filepath)
                        except Exception as err:
                            print("Error while encoding the confusion matrix plot",err)
                            raise Exception('Error while encoding the confusion matrix plot')
                        # End of ploting confusion matrix

                        # Plot the group accuracy matrix, save it in a file, encode it, put it in the html and delete the file
                        
                        group_accuracies = []
                        for category in np.unique(solution):
                            for s in np.unique(style):
                                group_index = np.where((solution==category) & (style==s))
                                group_accuracies.append(accuracy_score(solution[group_index],prediction[group_index]))
                        
                        group_accuracy_matrix=np.array(group_accuracies).reshape(len(np.unique(style)),len(np.unique(solution)))
                        fig, ax = plt.subplots(figsize=(7.5, 7.5))
                        ax.matshow(group_accuracy_matrix, cmap=plt.cm.Blues, alpha=0.3)
                        for i in range(group_accuracy_matrix.shape[0]):
                            for j in range(group_accuracy_matrix.shape[1]):
                                ax.text(x=j, y=i,s=group_accuracy_matrix[i, j], va='center', ha='center', size='xx-large')
                        plt.xlabel('Categories', fontsize=18)
                        plt.ylabel('Styles', fontsize=18)
                        plt.title('Group Accuracy Matrix', fontsize=18)
                        plt.savefig("group_accuracy_matrix.png")
                        try:
                            filepath="group_accuracy_matrix.png"
                            binary_fc = open(filepath, 'rb').read()
                            base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
                            ext = filepath.split('.')[-1]
                            dataurl = f'data:image/{ext};base64,{base64_utf8_str}'
                            html_file.write("Group Accuracy Matrix:")
                            html_file.write("<img src="+dataurl+" alt='Group Accuracy Matrix' width='250'/>")
                            html_file.write("<br>")
                            os.remove(filepath)
                        except Exception as err:
                            print("Error while encoding the confusion matrix plot",err)
                            raise Exception('Error while encoding the confusion matrix plot')
                        # End of ploting confusion matrix

                    except:
                        print("#--ERROR--# Error in calculation of the specific score of the task")
                        raise Exception('Error in calculation of the specific score of the task')

                    if debug_mode > 0:
                        scores = compute_all_scores(solution, prediction)
                        write_scores(html_file, scores)

                except Exception as inst:
                    score = missing_score
                    print(
                        "======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name + "): " + metric_name + "(" + score_name + ")=ERROR =======")
                    html_file.write(
                        "======= Set %d" % set_num + " (" + data_name.capitalize() + "_" + solution_name +  "): " + metric_name + "(" + score_name + ")=ERROR =======\n")
                    print("Error in scoring program: ", inst)

                # Write score corresponding to selected task and metric to the output file
                score_json[score_name] = score

            # End loop for solution_file in solution_names
            
            # Read the execution time and add it to the scores:
            try:
                metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
                score_json['duration'] = metadata['elapsedTime']
            except:
                score_json['duration'] = time.time() - start
            html_file.close()
            score_file.write(json.dumps(score_json))
            score_file.close()

            # Lots of debug stuff
            if debug_mode > 1:
                swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
                show_platform()
                show_io(prediction_dir, score_dir)
                show_version(scoring_version)
        except:
            pass
    
    # Read all the scores{i}.json files and write them in a single file by average the set1_score and set2_score
    scores = {"set1_score": 0.0, "set2_score": 0.0, "duration": 0.0}
    score_counter = 0
    for i in range(10):
        # Check if a file exists
        if not os.path.isfile(os.path.join(solution_dir, "tasks", f"labels{i}.csv")):
            continue
        try:
            with open(os.path.join(score_dir, f'scores{i}.json'), 'r') as f:
                scores_i = json.load(f)
                for k in scores:
                    scores[k] = scores[k] + scores_i[k]
            score_counter += 1
        except:
            pass

    if len(scores) == 0:
        raise Exception('No scores file found')
    # Average the scores
    for k in scores:
        scores[k] /= score_counter
    # Write the scores in a file
    with open(os.path.join(score_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f)


        
        
        
        
        
        