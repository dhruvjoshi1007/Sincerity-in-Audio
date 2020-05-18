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

	data_seq = []
	data_non_seq = []
	label = []
	shapee = np.array([])

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

			file = pd.read_csv('/home/helium-balloons/Desktop/midas/audio_features_egemaps_new/{}{}'.format(f[0].split('.')[0],'.csv'), sep=';')

			file = file.drop(columns=['name', 'frameTime'])
			file = file.to_numpy()


			padded = np.zeros([1500,130])
			padded[1500-file.shape[0]:,:file.shape[0]] = file




			row = lines[0:300]
			row = np.array(row)

			data_seq.append(padded)
			data_non_seq.append(row)
			label.append(1)
			
		except:
			print("------------ error in file -------- =   {}".format(f[0]))

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

			file = pd.read_csv('/home/helium-balloons/Desktop/midas/audio_features_egemaps_new/{}{}'.format(f[0].split('.')[0],'.csv'), sep=';')
			file = file.drop(columns=['name', 'frameTime'])
			file = file.to_numpy()

			padded = np.zeros([1500,130])
			padded[1500-file.shape[0]:,:file.shape[0]] = file

			

			row = lines[0:300]

			row = np.array(row)

			data_seq.append(padded)
			data_non_seq.append(row)
			label.append(0)
		except:
			print("------------ error in file -------- =   {}".format(f[0]))

	data_seq = np.array(data_seq)
	label = np.array(label)
	data_non_seq = np.array(data_non_seq)

	return data_seq, data_non_seq, label



### Train

path_train_S = '/home/helium-balloons/Desktop/midas/C-SIF/train-S.csv'
path_train_NS = '/home/helium-balloons/Desktop/midas/C-SIF/train-NS.csv'

data_seq, data_non_seq, label_seq = get_data(path_train_S, path_train_NS)

# print(data.shape)
# print(label.shape)


### Validation

path_val_S = '/home/helium-balloons/Desktop/midas/C-SIF/val-S.csv'
path_val_NS = '/home/helium-balloons/Desktop/midas/C-SIF/val-NS.csv'

val_data_seq, val_data_non_seq, val_label = get_data(path_val_S, path_val_NS)

### Test

path_test_S = '/home/helium-balloons/Desktop/midas/C-SIF/test-S.csv'
path_test_NS = '/home/helium-balloons/Desktop/midas/C-SIF/test-NS.csv'

test_data_seq, test_data_non_seq, test_label = get_data(path_test_S, path_test_NS)


##############3

def get_f1(y_true, y_pred): #taken from old keras source code
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	recall = true_positives / (possible_positives + K.epsilon())
	f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
	return f1_val

def recall_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


model = tf.keras.Sequential()
model.add(tf.keras.layers.Input([1500,130]))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True), input_shape=(1500, 130)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
# model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
# model.add(tf.keras.layers.Activation('sigmoid'))

input_layer = tf.keras.layers.Input([300])

mix = tf.keras.layers.concatenate([model.output, input_layer], axis=1)

dense1 = tf.keras.layers.Dense(64, activation='relu')(mix)

output = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)

model_final = tf.keras.models.Model(input = [model.input, input_layer], output = output)

print(data_seq.shape)
print('=============================================')

model_final.fit([data_seq, data_non_seq], epochs = 10)
