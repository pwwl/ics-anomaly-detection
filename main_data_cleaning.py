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

import pandas as pd 
import numpy as np
import pdb
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import kstest, ks_2samp

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from data_loader import load_train_data, load_test_data

def ecdf(samples, cutoff):
	return np.mean(samples < cutoff)

def compute_ks_star(samples1, samples2):

	# Compute range of integration
	start = min(np.min(samples1), np.min(samples1))
	end = max(np.max(samples2), np.max(samples2))
	integration_range = np.linspace(start, end, num=1000)

	ks_val = 0

	for xx in integration_range:
		ks_val += np.abs(ecdf(samples2, xx) - ecdf(samples1, xx))

	return ks_val

def clean_data(dataset_name):

	df_train = pd.read_csv("data/" + dataset_name + "/SWATv0_train.csv", dayfirst=True)
	df_test = pd.read_csv("data/" + dataset_name + "/SWATv0_test.csv")

	_, sensor_cols = load_train_data(dataset_name)
	_, Ytest, _ = load_test_data(dataset_name)

	Xtrain_raw = df_train[sensor_cols].values
	Xtest_raw = df_test[sensor_cols].values
	Xtest_raw_benign = Xtest_raw[Ytest == False]

	num_lost = 0
	ks_values = []

	for i in range(Xtrain_raw.shape[1]):

		ks_star = compute_ks_star(Xtrain_raw[:,i], Xtest_raw_benign[:,i])
		ks_values.append(ks_star)

		# Using the scipy library call
		# ks_result = ks_2samp(Xtrain_raw[:,i], Xtest_raw_benign[:,i])
		
		print(f'{sensor_cols[i]}: {ks_star}')

		if ks_star > 50:
			num_lost += 1

	print(f"Would lose {num_lost} out of {Xtrain_raw.shape[1]} features")
	print('OK')

if __name__ == '__main__':

	clean_data('SWAT')
