"""

   Copyright 2020 Lujo Bauer, Clement Fung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
# Uncomment section if EXACT reproducible results are needed
# rseed = 2021
# from numpy.random import seed
# seed(rseed)
# from tensorflow import set_random_seed
# set_random_seed(rseed)

# Generic python
import argparse
import pdb
import os
import sys
import json
import pickle
import time

# Data science ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import load_model

# Custom packages
from detector import autoencoder, lstm, cnn, dnn, gru, identity, linear
from data_loader import load_train_data, load_test_data

import metrics
import utils

def train_model(model_type, config, Xtrain, Xval, dataset_name=None):

    train_params = config['train']
    model_params = config['model']

    # define model input size parameter --- needed for AE size
    model_params['nI'] = Xtrain.shape[1]

    if model_type == 'AE':
        event_detector = autoencoder.AEED(**model_params)
    elif model_type == 'GRU':
        event_detector = gru.GatedRecurrentUnit(**model_params)
    elif model_type == 'LSTM':
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == 'DNN':
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == 'CNN':
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == 'LIN':
        event_detector = linear.Linear(**model_params)
    elif model_type == 'ID':
        event_detector = identity.Identity(**model_params)
    else:
        print(f'Model type {model_type} is not supported.')
        return

    event_detector.create_model()    
    event_detector.train(Xtrain,
            validation_data=(Xval, Xval),
            **train_params)

    return event_detector

def hyperparameter_search(event_detector, model_type, config, Xval, Xtest, Ytest, dataset_name, save=True, test_split=0.7, run_name='results', verbose=1):

    model_name = config['name']
    do_batches = False

    Ytest = Ytest.astype(int)
    Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils.custom_train_test_split(dataset_name, Xtest, Ytest, test_size=test_split, shuffle=False)
    #Xtest_val, Xtest_test, Ytest_val, Ytest_test = train_test_split(Xtest, Ytest, test_size=test_split, shuffle=False)

    if not model_type == 'AE':

        # Clip the prediction to match LSTM prediction window
        Ytest_test = Ytest_test[event_detector.params['history'] + 1:]
        Ytest_val = Ytest_val[event_detector.params['history'] + 1:]
        do_batches = True

    ##### Cross Validation
    validation_errors = event_detector.reconstruction_errors(Xval, batches=do_batches)
    test_errors = event_detector.reconstruction_errors(Xtest_val, batches=do_batches)
    test_instance_errors = test_errors.mean(axis=1)

    # Default to empty dict. Will still do F1 for window=1, ptile=95%
    grid_config = config.get('grid_search', dict())

    cutoffs = grid_config.get('percentile', [0.95])
    windows = grid_config.get('window', [1])
    eval_metrics = grid_config.get('metrics', ['F1'])

    for metric in eval_metrics:

        # FPR is a negative metric (lower is better)
        negative_metric = (metric == 'false_positive_rate')

        # FPR is a negative metric (lower is better)
        if negative_metric:
            best_metric = 1
        else:
            best_metric = -1000

        best_percentile = 0
        best_window = 0
        metric_vals = np.zeros((len(cutoffs), len(windows)))
        metric_func = metrics.get(metric)

        for percentile_idx in range(len(cutoffs)):

            percentile = cutoffs[percentile_idx]

            # set threshold as quantile of average reconstruction error
            theta = np.quantile(validation_errors.mean(axis = 1), percentile)

            for window_idx in range(len(windows)):

                window = windows[window_idx]

                #Yhat = event_detector.detect(Xtest, theta = theta, window = window, batches=True)
                Yhat = event_detector.cached_detect(test_instance_errors, theta = theta, window = window)
                #Yhat = Yhat[window-1:].astype(int)

                choice_value = metric_func(Yhat, Ytest_val)

                if verbose > 0:
                    print("{} is {:.3f} at theta={:.3f}, percentile={:.4f}, window={}".format(metric, choice_value, theta, percentile, window))

                # FPR is a negative metric (lower is better)
                if negative_metric:
                    if choice_value < best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window
                else:
                    if choice_value > best_metric:
                        best_metric = choice_value
                        best_percentile = percentile
                        best_window = window

                if grid_config.get('save-metric-info', False):
                    metric_vals[percentile_idx, window_idx] = choice_value

                if grid_config.get('detection-plots', False):

                    fig_detect, ax_detect = plt.subplots(figsize=(20, 4))

                    ax_detect.plot(Yhat, color = '0.1', label = 'predicted state')
                    ax_detect.plot(Ytest_val, color = 'r', alpha = 0.75, lw = 2, label = 'real state')
                    ax_detect.fill_between(np.arange(len(Yhat)), 0, Yhat.astype(int), color = '0.1')
                    ax_detect.set_title(
                        'Detection trajectory on test dataset, {}, percentile={:.3f}, window={}'
                        .format(model_type, percentile, window), fontsize = 14)
                    ax_detect.set_yticks([0,1])
                    ax_detect.set_yticklabels(['NO ATTACK','ATTACK'])
                    ax_detect.legend(fontsize = 12, loc = 2)
                    fig_detect.savefig(f'plots/{run_name}/{model_name}-{percentile}-{window}.pdf')
                    plt.close(fig_detect)

            if save:
                pickle.dump(theta, open(f'models/{run_name}/{model_name}-{percentile}-theta.pkl', 'wb'))
                print('Saved theta parameter to {0}.'.format('theta.pkl'))

        print("Best metric ({}) is {:.3f} at percentile={:.5f}, window {}".format(metric, best_metric, best_percentile, best_window))

        # Final test performance
        final_test_errors = event_detector.reconstruction_errors(Xtest_test, batches=do_batches)
        final_test_instance_errors = final_test_errors.mean(axis=1)

        best_theta = np.quantile(validation_errors.mean(axis = 1), best_percentile)
        event_detector.save_detection_params(best_theta=best_theta, best_window=best_window)

        final_Yhat = event_detector.best_cached_detect(final_test_instance_errors)
        #final_Yhat = final_Yhat[best_window-1:].astype(int)

        metric_func = metrics.get(metric)
        final_value = metric_func(final_Yhat, Ytest_test)
        print("Final {} is {:.3f} at percentile={:.5f}, window {}".format(metric, final_value, best_percentile, best_window))

        if grid_config.get('save-metric-info', False):
            np.save(f'npys/{run_name}/{model_name}-{metric}.npy', metric_vals)
            print(f'Saved metric at npys/{run_name}/{model_name}-{metric}.npy')

    return event_detector

def save_model(event_detector, config, run_name='results'):
    model_name = config['name']
    event_detector.save(f'models/{run_name}/{model_name}')

# functions
def load_saved_model(model_type, params_filename, model_filename):
    """ Load stored model. """

    if model_type == 'ID':
        return identity.Identity()

    # load params and create event detector
    with open(params_filename) as fd:
        model_params = json.load(fd)

    if model_type == 'AE':
        event_detector = autoencoder.AEED(**model_params)
    elif model_type == 'CNN':
        event_detector = cnn.ConvNN(**model_params)
    elif model_type == 'DNN':
        event_detector = dnn.DeepNN(**model_params)
    elif model_type == 'LSTM':
        event_detector = lstm.LongShortTermMemory(**model_params)
    elif model_type == 'LIN':
        event_detector = linear.Linear(**model_params)
    else:
        print(f'Model type {model_type} is not supported.')
        return None

    # load keras model
    event_detector.inner = load_model(model_filename)

    return event_detector

def parse_arguments():

    parser = utils.get_argparser()

    ### Train Params
    parser.add_argument("--train_params_epochs",
        default=100,
        type=int,
        help="Number of training epochs")
    parser.add_argument("--train_params_batch_size",
        default=512,
        type=int,
        help="Training batch size. Note: MUST be larger than history/window values given")
    parser.add_argument("--train_params_no_callbacks",
        action='store_true',
        help="Remove callbacks like early stopping")

    # Hyperparameters
    parser.add_argument("--detect_params_percentile",
        default=[0.95, 0.96, 0.97, 0.98, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.99995],
        nargs='+',
        type=float,
        help="Percentiles to look over")
    parser.add_argument("--detect_params_windows",
        default=[1, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        nargs='+',
        type=int,
        help="Windows to look over")
    parser.add_argument("--detect_params_metrics",
        default=['F1'],
        nargs='+',
        type=str,
        help="Metrics to look over")
    parser.add_argument("--detect_params_test_split",
        default=0.7,
        type=float,
        help="Split for testing/validation of detection hyperparameters. Default is 0.7 (hyperparameters evaluated on 30% of test data, final testing on 70%.) ")

    # saving items
    parser.add_argument("--detect_params_plots",
        action='store_true',
        help="Make detection plots for each hyperparameter setting")
    parser.add_argument("--detect_params_save_npy",
        action='store_true',
        help="Save the metric values in an npy")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

    # Define training parameters
    ae_train_params = {
        'verbose' : 1,
        'batch_size': args.train_params_batch_size,
        'epochs': args.train_params_epochs,
        'use_callbacks': not args.train_params_no_callbacks,
    }

    large_train_params = {
        'batch_size': args.train_params_batch_size,
        'epochs': args.train_params_epochs,
        'use_callbacks': not args.train_params_no_callbacks,
        'steps_per_epoch': 0,
        'validation_steps': 0,
        'verbose' : 1
    }

    config = {
        'grid_search': {
            'percentile': args.detect_params_percentile,
            'window': args.detect_params_windows,
            'metrics': args.detect_params_metrics,
            'pr-plot': False,
            'detection-plots': args.detect_params_plots,
            'save-metric-info': args.detect_params_save_npy
        }
    }

    run_name = args.run_name
    test_split = args.detect_params_test_split
    utils.update_config_model(args, config, model_type, dataset_name)
    model_name = config['name']

    Xtrain, Xval, sensor_cols = load_train_data(dataset_name, train_shuffle=True)
    Xtest, Ytest, _ = load_test_data(dataset_name)

    # Updates training parameters such as batch size, learning rate, etc.
    if model_type == 'AE':
        config.update({'train': ae_train_params})
    else:
        large_train_params['steps_per_epoch'] = len(Xtrain) // large_train_params['batch_size']
        large_train_params['validation_steps'] = len(Xval) // large_train_params['batch_size']
        config.update({'train': large_train_params})

    # Trains and returns the inner event detection model
    event_detector = train_model(model_type, config, Xtrain, Xval, dataset_name)
    save_model(event_detector, config, run_name=run_name)
    
    # Search for the best tuning of the window and theta parameters
    hyperparameter_search(event_detector, model_type, config, Xval, Xtest, Ytest, dataset_name,
        test_split=test_split,
        run_name=run_name,
        save=False,
        verbose=0)

    print("Finished!")
