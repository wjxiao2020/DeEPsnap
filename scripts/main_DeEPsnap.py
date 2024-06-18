# Copy right Weijia Xiao and Xue Zhang. All rights reserved.
# Authors: Weijia Xiao, Xue Zhang
# Northeastern University, Shaoyang University
# June 2024

import time
import os
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from DeEPsnap import DeEPsnap
from utils import make_folder, write_eval_stats, load_pickle


# configuring a logger
logging.basicConfig(format='%(levelname)s: %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('main')

result_dir = os.path.dirname(os.path.realpath(__file__)) + '/../results'
feature_dir = os.path.dirname(os.path.realpath(__file__)) + '/../features/'

# create folders needed to store the test results of the model
# assumes features folder already exists and contains data
make_folder(result_dir, logger)


# a dictionary storing the hyperparameters
params = {
    # the number of fold to use for K-fold cross validation
    'kfold' : 10,
    # the name of this experiment, will be the name of the record file
    'expName' : 'kv10-hl1024-weight4.5-drop02-hidden3_seq_emb_GO_complex_domain3'
}


def get_data():
    ''' Loads and prepares the data.
    Returns:
        X (numpy.ndarray) : all the features
        y (numpy.ndarray) : a column of labels
    '''
    fp1 = os.path.join(feature_dir, 'ess_seqF_embedF_GO_complex_domain3.pickle')
    fp2 = os.path.join(feature_dir, 'ness_seqF_embedF_GO_complex_domain3.pickle')
    ess_features = load_pickle(fp1)
    ness_features = load_pickle(fp2)
    data = np.vstack((ess_features, ness_features))
    
    X = data[:, : -1]
    y = data[:, -1]
    return X, y


def main():

    # program start time
    start_time = time.time()
    
    # creating a file to store evaluation statistics
    summary_record_file_path = os.path.join(result_dir, params['expName'] + '.tab')
    summary_record_file = open(summary_record_file_path, 'w')
    summary_record_file.write("Experiment Name: " + str(params['expName']) + '\n')
    summary_record_file.write("Fold\t" + "AUROC\t" + "AUPRC\t" + "Sensitivity\t" + "Specificity\t" + "PPV\t" + "Accuracy\t" +
                 "MCC\t" + "F1\t" + "BA\n")

    # a dictionary to store the evaluation statistics to calculate the average values
    eval_results_avg = {
        'roc_auc': 0.,
        'au_prc': 0.,
        'sensitivity': 0.,
        'specificity': 0.,
        'PPV': 0.,
        'accuracy': 0.,
        'MCC': 0.,
        'F1': 0.,
        'BA': 0.
    }
    
    # load data
    X,y = get_data()
    folds = StratifiedKFold(n_splits=params['kfold'], shuffle=True)
    fold = 0
    
    # perform K-fold cross validation
    for train_indices, test_indices in folds.split(X, y):
        fold += 1
        print('Fold ', fold)
        
        # split the data into training and testing set
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        # standardize the features using the mean and std in the training set
        # making all feature columns in the training set to have a mean of 0 and std of 1
        mean_X_train = X_train.mean(axis=0)
        std_X_train = X_train.std(axis=0)
        X_train = (X_train - mean_X_train) / std_X_train
        X_test = (X_test - mean_X_train) / std_X_train
        
        # pack into a list because the model class requires such an input type
        data = [X_train, y_train, X_test, y_test]
        
        logger.info('The size of training and validation data: {} * {}.'.format(X_train.shape[0], X_train.shape[1]))
        logger.info('The size of testing data: {} * {}.'.format(X_test.shape[0], X_test.shape[1]))
        
        # build, train, and test the DeEPsnap model
        model = DeEPsnap(data)
        eval_stats, roc_stats, prc_stats = model.get_eval_results()

        print(eval_stats)

        # record evaluation statistics into the file
        write_eval_stats(eval_stats, summary_record_file, fold)

        eval_results_avg['roc_auc'] += eval_stats['roc_auc']
        eval_results_avg['au_prc'] += eval_stats['au_prc']
        eval_results_avg['sensitivity'] += eval_stats['sensitivity']
        eval_results_avg['specificity'] += eval_stats['specificity']
        eval_results_avg['PPV'] += eval_stats['PPV']
        eval_results_avg['accuracy'] += eval_stats['accuracy']
        eval_results_avg['MCC'] += eval_stats['MCC']
        eval_results_avg['F1'] += eval_stats['F1']
        eval_results_avg['BA'] += eval_stats['BA']
        

    # average the summed evaluation results
    for value in eval_results_avg:
        eval_results_avg[value] = float(eval_results_avg[value]) / params['kfold']

    # record the averaged statistics across the folds, and some of the model hyperparameters
    write_eval_stats(eval_results_avg, summary_record_file, 'Avg.')
    summary_record_file.write("\n")
    summary_record_file.write('Batch size:' + str(eval_stats['batch_size']) + '\n')
    summary_record_file.write('Activation:' + str(eval_stats['activation']) + '\n')
    summary_record_file.write('Dropout:' + str(eval_stats['dropout']) + '\n')

    end_time = time.time()
    summary_record_file.write("Execution time: " + str(end_time - start_time) + " sec.")
    summary_record_file.close()
    

if __name__ == "__main__":
    main()

