# Copy right Weijia Xiao and Xue Zhang. All rights reserved.
# Authors: Weijia Xiao, Xue Zhang
# Northeastern University, Shaoyang University
# June 2024

import os
import pickle


def make_folder(folder_path, logger):
    ''' Creates a folder if it doesn't already exist.
    Args:
        folder_path (str) : the path of the folder to be created
        logger (logger) : a logger that will be used for logging
    '''
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        logger.info('{} is created.'.format(folder_path))
    else:
        logger.info('{} is already there.'.format(folder_path))


def write_eval_stats(eval_stats, record_file, iteration):
    ''' Writes the evaluation statistics into a file.
    Args:
        eval_stats (dict) : a dictionary recording all the evaluation statistics
        record_file (file) : a file object of the record file with write access
        iteration (int) : the current iteration number which will also be recorded
    '''
    record_file.write(str(iteration) + "\t" + str(eval_stats['roc_auc']) + "\t" +
                 str(eval_stats['au_prc']) + '\t' + str(eval_stats['sensitivity']) + '\t' +
                 str(eval_stats['specificity']) + '\t' + str(eval_stats['PPV']) + '\t' +
                 str(eval_stats['accuracy']) + '\t' + str(eval_stats['MCC']) + '\t' +
                 str(eval_stats['F1']) + '\t' + str(eval_stats['BA']) + '\n')
    
    
def load_pickle(fp):
    ''' Load data from a pickle file.
    Args:
        fp (str) : path of a pickle file
    Returns:
        data (any) : data stored in the pickle file
    '''
    with open(fp, 'rb') as pickle_in:
        data = pickle.load(pickle_in)
    return data