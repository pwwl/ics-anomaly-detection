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

# Custom packages
from main_train import load_saved_model
from data_loader import load_train_data, load_test_data

import metrics  
import utils                                  

def hyperparameter_search(event_detector, model_name, config, hp_metric, validation_errors, Xtestval_errors, Xtesttest_errors, Ytest_val, Ytest_test, run_name='results', verbose=1):

    # Default to empty dict. Will still do F1 for window=1, ptile=95%
    grid_config = config.get('grid_search', dict())

    cutoffs = grid_config.get('percentile', [0.95])
    windows = grid_config.get('window', [1])
    eval_metrics = grid_config.get('eval_metrics', ['F1'])

    test_instance_errors = Xtestval_errors.mean(axis=1)
    final_test_instance_errors = Xtesttest_errors.mean(axis=1)

    # FPR is a negative metric (lower is better)
    negative_metric = (hp_metric == 'FP')

    # FPR is a negative metric (lower is better)
    if negative_metric:
        best_metric = 1
    else:
        best_metric = -1000

    best_percentile = 0
    best_window = 0
    metric_vals = np.zeros((len(cutoffs), len(windows)))
    metric_func = metrics.get(hp_metric)

    for percentile_idx in range(len(cutoffs)):

        percentile = cutoffs[percentile_idx]

        # set threshold as quantile of average reconstruction error
        theta = np.quantile(validation_errors.mean(axis = 1), percentile)

        for window_idx in range(len(windows)):

            window = windows[window_idx]
            Yhat = event_detector.cached_detect(test_instance_errors, theta = theta, window = window)
            Yhat = Yhat[window-1:].astype(int)

            # Due to window sizes, need to normalize lengths before scoring
            Yhat_trunc, Ytest_trunc = utils.normalize_array_length(Yhat, Ytest_val)
            choice_value = metric_func(Yhat_trunc, Ytest_trunc)
            
            if verbose > 0:
                print("{} is {:.3f} at theta={:.3f}, percentile={:.4f}, window={}".format(hp_metric, choice_value, theta, percentile, window))

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

    best_theta = np.quantile(validation_errors.mean(axis = 1), best_percentile)
    print("Best metric ({}) is {:.3f} at theta={:.5f}, percentile={:.5f}, window {}".format(hp_metric, best_metric, best_theta, best_percentile, best_window))
    
    final_Yhat = event_detector.cached_detect(final_test_instance_errors, theta=best_theta, window=best_window)
    final_Yhat = final_Yhat[best_window-1:].astype(int)
    final_Yhat_trunc, final_Ytest_trunc = utils.normalize_array_length(final_Yhat, Ytest_test)

    final_value = metric_func(final_Yhat_trunc, final_Ytest_trunc)
    print("Final {} is {:.3f} at theta={:.5f}, percentile={:.5f}, window {}".format(hp_metric, final_value, best_theta, best_percentile, best_window))

    return best_percentile, best_window

def hyperparameter_eval(event_detector, model_name, config, validation_errors, Xtesttest_errors, Ytest_test, best_percentile, best_window, hp_metric, run_name='results', verbose=1, plot=False):

    # Default to empty dict. Will still do F1 for window=1, ptile=95%
    grid_config = config.get('grid_search', dict())
    eval_metrics = grid_config.get('eval_metrics', ['F1'])

    final_test_instance_errors = Xtesttest_errors.mean(axis=1)
    best_theta = np.quantile(validation_errors.mean(axis = 1), best_percentile)
    final_Yhat = event_detector.cached_detect(final_test_instance_errors, theta=best_theta, window=best_window)
    final_Yhat = final_Yhat[best_window-1:].astype(int)

    final_values = []

    for metric in eval_metrics:
        
        metric_func = metrics.get(metric)

        # Due to window sizes, need to normalize lengths before scoring
        final_Yhat_trunc, Ytest_trunc = utils.normalize_array_length(final_Yhat, Ytest_test)
        final_value = metric_func(final_Yhat_trunc, Ytest_trunc)
        print("Final {} is {:.3f} at percentile={:.5f}, window {}".format(metric, final_value, best_percentile, best_window))

        final_values.append({metric :final_value})

    if plot:

        fig, ax = plt.subplots(figsize=(20, 4))

        ax.plot(-1 * final_Yhat, color = '0.25', label = 'Predicted')
        ax.plot(Ytest_test, color = 'lightcoral', alpha = 0.75, lw = 2, label = 'True Label')
        ax.fill_between(np.arange(len(final_Yhat)), -1 * final_Yhat, 0, color = '0.25')
        ax.fill_between(np.arange(len(Ytest_test)), 0, Ytest_test, color = 'lightcoral')
        ax.set_yticks([-1,0,1])
        ax.set_yticklabels(['Predicted','Benign','Attacked'])
        ax.set_title(f'Detection trajectory, best percentile={best_percentile}, best window={best_window}', fontsize = 36)
        fig.tight_layout()
        try:
            plt.savefig(f'plots/{run_name}/{model_name}-{best_percentile}-{best_window}.pdf')
        except FileNotFoundError:
            plt.savefig(f'plots/results/{model_name}-{best_percentile}-{best_window}.pdf')
            print(f"Unable to find plots/{run_name}/, saved {model_name}-{best_percentile}-{best_window}.pdf to plots/results/ instead")
        plt.close()

        plot_obj = []
        plot_obj.append(final_Yhat)
        plot_obj.append(Ytest_test)

        print(f'Dumping pkl object for {hp_metric}: {best_percentile} {best_window}')
        try:
            pickle.dump(plot_obj, open(f'outputs/{run_name}/{model_name}-{hp_metric}-{best_percentile}-{best_window}.pkl', 'wb'))
            print(f'Saved {model_name}-{hp_metric}-{best_percentile}-{best_window}.pkl in outputs/{run_name}/')
        except FileNotFoundError:
            pickle.dump(plot_obj, open(f'outputs/results/{model_name}-{hp_metric}-{best_percentile}-{best_window}.pkl', 'wb'))
            print(f"Unable to find outputs/{run_name}/, saved {model_name}-{hp_metric}-{best_percentile}-{best_window}.pkl to outputs/results/ instead")

    return final_values

def parse_arguments():
    
    parser = utils.get_argparser()

    # Detection hyperparameter search
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

    # Choice of metric
    parser.add_argument("--detect_params_hp_metrics", 
        default=['F1'],
        nargs='+',
        type=str,
        help="Metrics to look over")
    parser.add_argument("--detect_params_eval_metrics", 
        default=['F1'],
        nargs='+',
        type=str,
        help="Metrics to look over")
    parser.add_argument("--detect_params_test_split", 
        default=0.7,
        type=float,
        help="Split for testing/validation of detection hyperparameters. Default is 0.7 (hyperparameters evaluated on 30%% of test data, final testing on 70%%.) ")
    parser.add_argument("--eval_plots",
        action="store_true",
        help="Make detection plots for hyperparameter settings")

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()
    model_type = args.model
    dataset_name = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

    config = {
        'grid_search': {
            'percentile': args.detect_params_percentile,
            'window': args.detect_params_windows,
            'eval_metrics': args.detect_params_eval_metrics,
        }
    }

    run_name = args.run_name
    test_split = args.detect_params_test_split
    utils.update_config_model(args, config, model_type, dataset_name)
    model_name = config['name']
    hp_metrics = args.detect_params_hp_metrics

    Xfull, sensor_cols = load_train_data(dataset_name, train_shuffle=True)
    Xtest, Ytest, _ = load_test_data(dataset_name)
    Ytest = Ytest.astype(int)
    event_detector = load_saved_model(model_type, run_name, model_name)

    do_batches = False
    Xtest_val, Xtest_test, Ytest_val, Ytest_test = utils.custom_train_test_split(dataset_name, Xtest, Ytest, test_size=test_split, shuffle=False)

    if not model_type == 'AE':
        
        # Clip the prediction to match prediction window
        history = config['model']['history']
        Ytest_test = Ytest_test[history + 1:]
        Ytest_val = Ytest_val[history + 1:]
        do_batches = True

        all_idxs = np.arange(history, len(Xfull)-1)
        _, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=0.2, random_state=42, shuffle=True)
        validation_errors = utils.reconstruction_errors_by_idxs(event_detector, Xfull, val_idxs, history)
    
    else:

        _, Xval, _, _  = train_test_split(Xfull, Xfull, test_size=0.2, random_state=42, shuffle=True)
        validation_errors = event_detector.reconstruction_errors(Xval, batches=do_batches)

    Xtestval_errors = event_detector.reconstruction_errors(Xtest_val, batches=do_batches)

    # Final test performance
    Xtesttest_errors = event_detector.reconstruction_errors(Xtest_test, batches=do_batches)

    overall_values = []

    for hp_metric in hp_metrics:

        # Note: For models to be used in explanations, change the hyperparameter evaluation to use more data (such as test_split=0.01) for cleaner results
        # Search for the best tuning of the window and theta parameters
        bestp, bestw = hyperparameter_search(event_detector, model_type, config, 
            hp_metric,
            validation_errors,
            Xtestval_errors,
            Xtesttest_errors, 
            Ytest_val,
            Ytest_test,
            run_name=run_name, 
            verbose=0)

        final_values = hyperparameter_eval(event_detector, model_type, config, 
            validation_errors,
            Xtesttest_errors,
            Ytest_test,
            bestp,
            bestw,
            hp_metric,
            run_name=run_name,
            plot=args.eval_plots, 
            verbose=0)

        overall_values.append({hp_metric : final_values})

    try:
        np.save(f'outputs/{run_name}/{model_name}-model-tuning-scores.npy', overall_values)
        print(f'Saved output to {run_name}/{model_name}-model-tuning-scores.npy')
    except FileNotFoundError:
        np.save(f'outputs/results/{model_name}-model-tuning-scores.npy', overall_values)
        print(f"Unable to find outputs/{run_name}/, saved {model_name}-model-tuning-scores.npy to outputs/results/ instead")
        print(f"Note: we recommend creating outputs/{run_name}/ to store this output")

    print("Finished!")
