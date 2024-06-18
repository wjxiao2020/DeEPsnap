# Copy right Weijia Xiao and Xue Zhang. All rights reserved.
# Authors: Weijia Xiao, Xue Zhang
# Northeastern University, Shaoyang University
# June 2024

import os
import numpy as np
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import Callback, LearningRateScheduler
from math import sqrt, pi, sin
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, confusion_matrix
from utils import make_folder


# configuring a logger
logging.basicConfig(format='%(levelname)s: %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger('DeEPsnap')

# create folders needed to store the saved models and graph related informations
# assumes the Python scripts is stored in a different and parallel folder from those folders
model_dir = os.path.dirname(os.path.realpath(__file__)) + '/../models'
make_folder(model_dir, logger)


# a dictionary storing the hyperparameters for the DeEPsnap model
params = {
    # the total number of epochs the model will be trained
    'epoch' : 50,
    'batchSize' : 32,
    'dropout' : 0.3,
    'loss' : 'binary_crossentropy',
    'metrics': ['accuracy', 'auc'],
    # the initial learning rate at the start of each snapshot cycle
    'initial_lr' : 0.001,
    # the number of epochs of a full snapshot cycle, the model will be saved after every cycle
    'epoch_per_cycle' : 5,
    # the activation function to be used after each hidden layer
    'activation1' : 'relu',
    # the activation function to be used after the output layer
    'activation2' : 'sigmoid',
    # the percentage of the training set that will be used for validation
    'val_split' : 0,
    # setting the class weight to deal with class imbalance
    # (the number of the label 0 samples is around 4.5 times more than that of the label 1 samples)
    'class_weight' : {0: 1.0, 1: 4.5},
    # defines a list of the input shapes of all the hidden layers to be used in the DeEPsnap
    'hidden_layers' : [1024, 512, 256]
}


def snapshot_lr(epoch, lr):
    ''' Returns snapshot learning rate scheduling using the sine function.
    Args:
        epoch (int) : the current training epoch number (0 for first epoch)
        lr (float) : the current learning rate (not used during the calculation of the new learning rate,
                     just being here because LearningRateScheduler requires it)
    Returns:
        new_lr (float) : the new learning rate to be used for the current training epoch
    '''
    scaler = sin(pi * (1 + epoch % params['epoch_per_cycle'] / (params['epoch_per_cycle'] * 2))) + 1
    return scaler * params['initial_lr']



class DeEPsnap(object):

    def __init__(self, data, numHidden=len(params['hidden_layers'])):
        ''' Initialize, train, and test the DeEPsnap model.
        Args:
            data (list) : a list of input data in the order of [X_train, y_train, X_test, y_test]
            numHidden (int) : the number of hidden layers that the baseline MLP model should have
        '''
        super(DeEPsnap, self).__init__()
        
        self.X_train, self.y_train, self.X_test, self.y_test = data
        self.numHidden = numHidden
        
        # keep track of how many models are saved
        self.num_models_saved = 0
        
        self.model = self.build_model()
        self.eval_info, self.roc_stats, self.prc_stats = self.run_model()
        
    def get_eval_results(self):
        ''' Returns the evaluation statistics of the model.
        Returns:
            eval_info (dict) : a dictionary containing the results of all evaluation metrics on the model
                               and some model hyperparameters to record
            roc_stats (str) : a string of 3 lines representing the false positive rate, true positive rate,
                              and thresholds for ROC curve plot repectively
            prc_stats (str) : a string of 3 lines representing the precision, recall, and thresholds
                              for PR curve plot repectively
        '''
        return self.eval_info, self.roc_stats, self.prc_stats
    
    def get_snapshot_path(self):
        ''' Returns a file path for the current model to be saved during snapshot cycle.
        Returns:
            file_path (str) : a file path for the current model to be saved during snapshot cycle
                              where the recorded model id in the title increments from 0
        '''
        
        file_path = os.path.join(model_dir, 'snapshot_{}.weights.h5'.format(self.num_models_saved))
        self.num_models_saved += 1
        return file_path
    
    def build_model(self):
        ''' Builds the baseline MLP model structure.
        Returns:
            model (Sequential) : a Keras Sequential object containing the baseline MLP model structure
                                 with the configurations all set up and ready to be trained
        '''
        logger.info('The number of hidden layers: {}.'.format(self.numHidden))
        
        num_features = self.X_train.shape[1]
        model = Sequential()
        model.add(Input(shape=(num_features,)))
        for i in range(self.numHidden):
            model.add(Dense(params['hidden_layers'][i], activation = params['activation1']))
            model.add(Dropout(params['dropout']))
            
        # the final layer will output a single number between 0 and 1
        # representing the probability of the inputed gene is essential
        model.add(Dense(1, activation=params['activation2']))

        model.compile(optimizer=Adam(),
                      loss=params['loss'],
                      metrics=params['metrics'])
        return model
    
    def run_model(self):
        ''' Trains the model and make prediction on the test data.
        Returns:
            eval_info (dict) : a dictionary containing the results of all evaluation metrics on the model
                               and some model hyperparameters to record
            roc_stats (str) : a string of 3 lines representing the false positive rate, true positive rate,
                              and thresholds for ROC curve plot repectively
            prc_stats (str) : a string of 3 lines representing the precision, recall, and thresholds
                              for PR curve plot repectively
        '''
        # create a check point to save the model weights at the end of each cycle during training
        snapshot_checkpoint = SnapshotCheckpoint(
                                    save_freq = params['epoch_per_cycle'],
                                    model_instance = self
                              )
                              
        # create the LearningRateScheduler callback for a snapshot training cycle
        snapshot_lr_scheduler = LearningRateScheduler(snapshot_lr)

        # fit the model to the training data and verify with validation data if any
        self.model.fit(self.X_train, self.y_train,
                  epochs=params['epoch'],
                  callbacks=[snapshot_lr_scheduler, snapshot_checkpoint],
                  batch_size=params['batchSize'],
                  shuffle=True, verbose=1, validation_split=params['val_split']
                  , class_weight = params['class_weight'])

        assert self.num_models_saved > 0, "\nWarning: No model saved!"
        
        # load each saved model and sum up their predictions
        for model_id in range(self.num_models_saved):
            model_saved_path = os.path.join(model_dir, 'snapshot_{}.weights.h5'.format(model_id))
            self.model.load_weights(model_saved_path)
            if model_id == 0:
                prediction = self.model.predict(self.X_test)
            else:
                prediction += self.model.predict(self.X_test)
        
        # average the predicted probability
        pred_prob = prediction / self.num_models_saved
        
        return self.evaluate_model(pred_prob)
        
    def evaluate_model(self, y_pred_prob):
        ''' Evaluate the performance of the model on the test data.
        Args:
            y_pred_prob (numpy.ndarray) : a column of probabilities predict on the test data
                                          of whether each gene is essential
        Returns:
            eval_info (dict) : a dictionary containing the results of all evaluation metrics on the model
                               and some model hyperparameters to record
            roc_stats (str) : a string of 3 lines representing the false positive rate, true positive rate,
                              and thresholds for ROC curve plot repectively
            prc_stats (str) : a string of 3 lines representing the precision, recall, and thresholds
                              for PR curve plot repectively
        '''
        y_pred_prob_one_hot = np.hstack((1 - y_pred_prob, y_pred_prob))
        y_test_one_hot = to_categorical(self.y_test)
        
        # calculate the AUROC score (area under the Receiver Operating Characteristic curve)
        roc_auc = roc_auc_score(y_test_one_hot, y_pred_prob_one_hot)
        
        # record the false positive rate, true positive rate, and thresholds for ROC curve plot
        fpr, tpr, threshold = roc_curve(self.y_test, y_pred_prob)
        roc_stats = '\t'.join(map(str, fpr)) + '\n' \
                    + '\t'.join(map(str, tpr)) + '\n' \
                    + '\t'.join(map(str, threshold)) + '\n'
        
        # calculate the AUPRC score (area under the precision-recall curve)
        au_prc = average_precision_score(y_test_one_hot, y_pred_prob_one_hot)
        
        # record the precision, recall, and thresholds for PR curve plot
        pre, rec, thr = precision_recall_curve(self.y_test, y_pred_prob)
        prc_stats = '\t'.join(map(str, rec)) + '\n' \
                    + '\t'.join(map(str, pre)) + '\n' \
                    + '\t'.join(map(str, thr)) + '\n'
        
        # get 1 column of integers representing the final predicted label
        y_pred = np.argmax(y_pred_prob_one_hot, axis=1)

        # get True Negative, False Positive, False Negative, True Positive
        TN, FP, FN, TP = confusion_matrix(self.y_test, y_pred).ravel()
        
        # PPV = precision, sensitivity = recall
        sensitivity = float(TP) / float(TP + FN)
        specificity = float(TN) / float(TN + FP)
        PPV = float(TP) / float(TP + FP)
        accuracy = float(TP + TN) / float(TP + FP + FN + TN)
        mcc = (TP * TN - FP * FN)/ sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        F1_measure = 2 * PPV * sensitivity/(PPV + sensitivity)
        BA = (sensitivity + specificity)/2
        
        # a dictionary to store evaluation stats and model parameters
        eval_info = {
            'roc_auc': roc_auc,
            'au_prc': au_prc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'PPV': PPV,
            'accuracy': accuracy,
            'MCC': mcc,
            'F1': F1_measure,
            'BA' : BA,
            'batch_size': params['batchSize'],
            'activation': params['activation2'],
            'dropout': params['dropout']
        }

        return eval_info, roc_stats, prc_stats
    
    

class SnapshotCheckpoint(Callback):
    '''
    Snapshot checkpoint callback to save model at the end of each cycle.
    '''
    def __init__(self, save_freq, model_instance):
        ''' Initialize a model checkpoint to be used during training.
        Args:
            save_freq (int) : the number of epochs that a snapshot cycle should have
            model_instance (DeEPsnap) : the DeEPsnap model instance that will be trained
        '''
        super(SnapshotCheckpoint, self).__init__()
        self.save_freq = save_freq
        self.model_instance = model_instance

    def on_epoch_end(self, epoch, logs=None):
        ''' Save the model weights if this is the end of a snapshot cycle.
        Args:
            epoch (int) : the current epoch number (0 for first epoch)
            logs (dict) : a dictionary containing attributes and some current information of the model
        '''
        if (epoch + 1) % self.save_freq == 0:
            filepath = self.model_instance.get_snapshot_path()
            self.model.save_weights(filepath)
            print("\nEpoch {} : Weights saved to {}".format(epoch + 1, filepath))

