import numpy as np
import librosa
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import glob

from keras.callbacks import Callback,ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import optimizers

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K


def get_data(path_S, path_NS):

	filenames_S = pd.read_csv(path_S)
	filenames_NS = pd.read_csv(path_NS)
	filenames_S = filenames_S.to_numpy()
	filenames_NS = filenames_NS.to_numpy()
	print(filenames_S.shape)
	# print(filenames_S)

	data = []
	label = []

	for f in filenames_S:
		# print(f)
		try:
		# file = pd.read_csv('/home/helium-balloons/Desktop/midas/audio_file/{}{}'.format(f[0],'.csv'))
			data_file = '/home/helium-balloons/Desktop/midas/audio_file/{}{}'.format(f[0],'.csv')
			with open(data_file, 'r') as temp_f:
			# Read the lines
				lines = temp_f.readlines()
				lines = lines[-1].split(',')
				lines = [float(x) for x in lines[1:len(lines)-1]]

			# row = lines[0:300]
			row = np.array(lines[0:700])
			# row2 = np.array(lines[1000:1200])
			# row = np.concatenate([row,row2])

			data.append(row)
			label.append(1)
			
		except Exception as e:
			print("------------ error in {} -------- =  {}".format(f[0], e))

	# shapee  = np.array([])

	for f in filenames_NS:

		try:
			data_file = '/home/helium-balloons/Desktop/midas/audio_file/{}{}'.format(f[0],'.csv')
			with open(data_file, 'r') as temp_f:
			# Read the lines
				lines = temp_f.readlines()
				lines = lines[-1].split(',')
					# print(lines)
				lines = [float(x) for x in lines[1:len(lines)-1]]
			

			# row = lines[0:300]
			row = np.array(lines[0:700])
			# row2 = np.array(lines[1000:1200])
			# row = np.concatenate([row,row2])

			data.append(row)
			label.append(0)
		except Exception as e:
			print("------------ error in {} -------- =  {}".format(f[0], e))

	data = np.array(data)
	label = np.array(label)

	return data, label



### Train

path_train_S = '/home/helium-balloons/Desktop/midas/C-SIF/train-S.csv'
path_train_NS = '/home/helium-balloons/Desktop/midas/C-SIF/train-NS.csv'

data, label= get_data(path_train_S, path_train_NS)

# print(data.shape)
# print(label.shape)


### Validation

path_val_S = '/home/helium-balloons/Desktop/midas/C-SIF/val-S.csv'
path_val_NS = '/home/helium-balloons/Desktop/midas/C-SIF/val-NS.csv'

val_data, val_label = get_data(path_val_S, path_val_NS)

### Test

path_test_S = '/home/helium-balloons/Desktop/midas/C-SIF/test-S.csv'
path_test_NS = '/home/helium-balloons/Desktop/midas/C-SIF/test-NS.csv'

test_data, test_label = get_data(path_test_S, path_test_NS)

data = np.concatenate([data, val_data])
print(data.shape)
label = np.concatenate([label, val_label])
print(label.shape)

print('begin')

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
# clf = SVC(gamma='auto')

clf = SVC(kernel='rbf', gamma=1/1000, C=1, decision_function_shape='ovr') #gamma = 1/900

clf.fit(data, label)

print(clf.score(test_data, test_label, sample_weight=None))

y_pred = clf.predict(test_data)

print('f1_score')
print(f1_score(test_label, y_pred))

print('uar')
print(recall_score(test_label, y_pred, average='macro'))