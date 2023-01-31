'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import numpy as np
import scipy as sp
from sklearn.metrics import matthews_corrcoef,balanced_accuracy_score,f1_score,confusion_matrix,roc_auc_score,accuracy_score,precision_score,recall_score
from scipy.stats.mstats import gmean
def matthews_corrcoef_metric(solution, prediction, style_solution):
    # print(solution, prediction, style)
    '''Matthews correlation coefficient.
    Works even if the target matrix has more than one column'''
    score = matthews_corrcoef(solution,prediction)
    return np.mean(score)

def worst_group_accuracy_metric(solution, prediction, style_solution):
    '''Worst group accuracy.
    Works even if the target matrix has more than one column'''
    group_accuracies = []
    for category in np.unique(solution):
        for style in np.unique(style_solution):
            group_index = np.where((solution==category) & (style_solution==style))
            group_accuracies.append(accuracy_score(solution[group_index],prediction[group_index]))
    score = np.min(group_accuracies)
    return score

def geometric_mean_accuracy_metric(solution, prediction, style_solution):
    '''Worst group accuracy.
    Works even if the target matrix has more than one column'''
    group_accuracies = []
    for category in np.unique(solution):
        for style in np.unique(style_solution):
            group_index = np.where((solution==category) & (style_solution==style))
            group_accuracies.append(accuracy_score(solution[group_index],prediction[group_index]))
    print(group_accuracies)
    score = gmean(group_accuracies)
    return score
