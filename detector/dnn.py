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

# numpy stack
import json
import numpy as np
import pdb
import tensorflow as tf

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, MaxPool1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from .detector import ICSDetector

# ### RNN classes
# classes
class DeepNN(ICSDetector):
    """ Keras-based DNN class used for event detection.

        Attributes:
        params: dictionary with parameters defining the RNN structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor, stores parameters and initialize RNN Keras model. """

        # Default parameters values. If nI is not given, the code will crash later.
        params = {
            'nI': None,
            'units': 64,
            'history': 0,
            'layers': 3,
            'activation' : 'relu',
            'optimizer' : 'adam',
            'verbose' : 0
            }

        for key,item in kwargs.items():
            params[key] = item

        self.params = params

    def create_model(self):
        """ Creates Keras RNN model.
        """

        # retrieve params
        nI = self.params['nI'] # number of inputs
        units = self.params['units'] # number of units per layer
        layers = self.params['layers'] # Number of hidden layers
        activation = self.params['activation'] # Activation function between layers
        optimizer = self.params['optimizer'] # Keras optimizer
        verbose = self.params['verbose'] # echo on screen

        if layers < 1:
            print('Error: Must have at least one layer. Found layers={}'.format(layers))
            return

        input_layer = Input(shape=(nI))

        deep_layer = Dense(units, activation=activation)(input_layer)

        # Add any additional layers beyond 1.
        for x in range(layers - 1):
            deep_layer = Dense(units, activation=activation)(deep_layer)
        
        flatten = Flatten()(deep_layer)
        dense_out = Dense(nI)(flatten)
        
        # Define the total model
        model = Model(input_layer, dense_out)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        if verbose:
            print(model.summary())

        # compile and return model
        self.inner = model
        return model

    def transform_to_window_data(self, dataset, target, target_size=1):
        data = []
        labels = []

        start_index = 0
        end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            data.append(dataset[i])
            labels.append(target[i+target_size])

        return np.array(data), np.array(labels)

    def train(self, x, use_callbacks=False, **train_params):
        """ Train DNN,

            x: inputs (inputs == targets, We are training a self-supervised CNN).
        """

        if self.inner == None:
            print('Creating model.')
            self.create_model()

        if 'batch_size' not in train_params:
            batch_size = 32 
        else:
            # A bit hacky, since we have to manually do the batching for CNN/LSTM.
            batch_size = train_params['batch_size']
            del train_params['batch_size']

        # Generic data generator object for feeding data to fit_generator
        def data_generator(X, bs):
            
            i = 0
            while True:
                i += bs

                # Restart from beginning
                if i + bs > len(X):
                    i = 0 

                X_window, y_window = self.transform_to_window_data(X[i:i+bs], X[i:i+bs])
                yield (X_window, y_window)

        if use_callbacks:
            train_params['callbacks'] = [
                EarlyStopping(monitor='val_loss', patience=3, verbose=0,  min_delta=0, mode='auto')
            ]

        if 'validation_data' in train_params:
            
            x_val = train_params['validation_data'][0]
            train_params['validation_data'] = data_generator(x_val, batch_size)

        train_history = self.inner.fit(data_generator(x, batch_size), **train_params)
        
        # Save losses to CSV
        if self.params['verbose'] > 0:        
            loss_obj = np.vstack([train_history.history['loss'], train_history.history['val_loss']])
            np.savetxt(f'dnn-train-history-{self.params["layers"]}l-{self.params["units"]}u.csv', loss_obj, delimiter=',', fmt='%.5f')



    def detect(self, x, theta, window = 1, batches=False, eval_batch_size = 4096, **keras_params):
        """ Detection performed based on (smoothed) reconstruction errors.

            x = inputs,
            theta = threshold, attack flagged if reconstruction error > threshold,
            window = length of the smoothing window (default = 1 timestep, i.e. no smoothing),
            average = boolean (default = False), if True the detection is performed
                on the average reconstruction error across all outputs,
            keras_params = parameters for the Keras-based AE prediction.
        """
        reconstruction_error = self.reconstruction_errors(x, batches, eval_batch_size, **keras_params)
        
        # Takes the mean error over all features
        instance_errors = reconstruction_error.mean(axis=1)
        return self.cached_detect(instance_errors, theta, window)

    def cached_detect(self, instance_errors, theta, window = 1):
        """
            Same as detect, but using the errors pre-computed
        """

        # Takes the mean error over all features
        detection = instance_errors > theta

        # If window exceeds one, look for consective detections
        if window > 1:

            detection = np.convolve(detection, np.ones(window), 'same') // window

            # clement: Removing this behavior
            # Backfill the windows (e.g. if idx 255 is 1, all of 255-window:255 should be filled)
            # fill_idxs = np.where(detection)
            # fill_detection = detection.copy()

            # # Fill for both the history and the window
            # fill_amount = window + self.params['history']
            # for idx in fill_idxs[0]:
            #     fill_detection[idx - window : idx] = 1
            # return fill_detection

        return detection

    def reconstruction_errors(self, x, batches=False, eval_batch_size = 4096, **keras_params):
        
        if batches:
            
            full_errors = np.zeros((x.shape[0] - 1, x.shape[1]))
            idx = 0
            
            while idx < len(x):
                
                Xwindow, Ywindow = self.transform_to_window_data(x[idx: idx + eval_batch_size + 1], x[idx:idx + eval_batch_size + 1])

                if idx + eval_batch_size > len(full_errors):
                    full_errors[idx:] = (self.predict(Xwindow, **keras_params) - Ywindow)**2                
                else:
                    full_errors[idx:idx+eval_batch_size] = (self.predict(Xwindow, **keras_params) - Ywindow)**2
                
                idx += eval_batch_size

            return full_errors

        else:
            # CNN needs windowed data
            Xwindow, Ywindow = self.transform_to_window_data(x, x)
            return (self.predict(Xwindow, **keras_params) - Ywindow)**2


if __name__ == "__main__":
    print("Not a main file.")