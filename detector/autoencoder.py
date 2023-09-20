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

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers

from .detector import ICSDetector

# ### Autoencoder classes
# classes
class AEED(ICSDetector):
    """ Keras-based AutoEncoder (AE) class used for event detection.

        Attributes:
        params: dictionary with parameters defining the AE structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor, stores parameters and initialize AE Keras model. """

        # Default parameters values. If nI is not given, the code will crash later.
        params = {
            'nI': None,
            'nH': 5,
            'cf': 2,
            'activation' : 'relu',
            'optimizer' : 'adam',
            'verbose' : 0
            }

        for key,item in kwargs.items():
            params[key] = item

        self.params = params

    def create_model(self):
        """ Creates Keras AE model.

            The model has nI inputs, nH hidden layers in the encoder (and decoder)
            and cf compression factor. The compression factor is the ratio between
            the number of inputs and the innermost hidden layer which stands between
            the encoder and the decoder. The size of the hidden layers between the
            input (output) layer and the innermost layer decreases (increase) linearly
            according to the cg.
        """

        # retrieve params
        nI = self.params['nI'] # number of inputs
        nH = self.params['nH'] # number of hidden layers in encoder (decoder)
        cf = self.params['cf'] # compression factor
        activation = self.params['activation'] # autoencoder activation function
        optimizer = self.params['optimizer'] # Keras optimizer
        verbose = self.params['verbose'] # echo on screen

        # get number/size of hidden layers for encoder and decoder
        temp = np.linspace(nI,nI/cf,nH + 1).astype(int)
        nH_enc = temp[1:]
        nH_dec = temp[:-1][::-1]

        # input layer placeholder
        autoencoder = Sequential()

        input_layer = Input(shape=(nI,))

        # build encoder
        for i, layer_size in enumerate(nH_enc):
            if i == 0:
                autoencoder.add(Dense(layer_size, activation=activation, input_shape=(nI,)))
            else:
                # other hidden layers
                autoencoder.add(Dense(layer_size, activation=activation))

        # build decoder
        for i, layer_size in enumerate(nH_dec):
            if i == nH - 1:
                # For last output layer, don't use activation function.
                autoencoder.add(Dense(layer_size))
            else:
                # other hidden layers
                autoencoder.add(Dense(layer_size, activation=activation))

        # print autoencoder specs
        if verbose > 0:
            print('Created autoencoder with structure:');
            print(', '.join('layer_{}: {}'.format(v, i) for v, i in enumerate(np.hstack([nI,nH_enc,nH_dec]))))
            print(autoencoder.summary())

        # compile and return model
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        self.inner = autoencoder
        return autoencoder

    def train(self, x, use_callbacks=False, **train_params):
        """ Train autoencoder,

            x: inputs (inputs == targets, AE are self-supervised ANN).
        """
        
        if self.inner == None:
            print('Creating model.')
            self.create_model()
        
        # train models with early stopping and reduction of learning rate on plateau
        if use_callbacks:
            train_params['callbacks'] = [
                EarlyStopping(monitor='val_loss', patience=3, verbose=0,  min_delta=0, mode='auto')
            ]

        train_history = self.inner.fit(x, x, **train_params)
        
        # Save losses to CSV
        if self.params['verbose'] > 0:        
            loss_obj = np.vstack([train_history.history['loss'], train_history.history['val_loss']])
            np.savetxt(f'ae-train-history-{self.params["nH"]}l-{self.params["cf"]}u.csv', loss_obj, delimiter=',', fmt='%.5f')

    def detect(self, x, theta, window = 1, batches=False, **keras_params):
        """ Detection performed based on (smoothed) reconstruction errors.

            x = inputs,
            theta = threshold, attack flagged if reconstruction error > threshold,
            window = length of the smoothing window (default = 1 timestep, i.e. no smoothing),
            average = boolean (default = False), if True the detection is performed
                on the average reconstruction error across all outputs,
            keras_params = parameters for the Keras-based AE prediction.
        """
        #        preds = super(AEED, self).predict(x,keras_params)
        preds = self.predict(x, **keras_params)
        reconstruction_error = (x-preds)**2
        
        # Takes the mean error over all features
        instance_errors = reconstruction_error.mean(axis=1)
        return self.cached_detect(instance_errors, theta, window)

    def cached_detect(self, instance_errors, theta, window = 1):
        """
            Same as detect, but using the errors pre-computed
        """
        detection = instance_errors > theta

        # If window exceeds one, look for consecutive detections
        if window > 1:
            detection = np.convolve(detection, np.ones(window), 'same') // window

        return detection

    def reconstruction_errors(self, x, batches=False):
        return (self.predict(x) - x)**2

if __name__ == "__main__":
    print("Not a main file.")