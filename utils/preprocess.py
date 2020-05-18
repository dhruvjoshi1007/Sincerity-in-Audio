
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
				# print(lines)
				lines = [float(x) for x in lines[1:len(lines)-1]]


			# print(file.shape)
			# file = file.drop(columns=['name', 'frameTime'])
			# print(file.shape)
			# file = file.to_numpy()

			# print('-----------')
			# print(file.shape[0])
			# print()
			row = lines[0:300]
			# row = file.iloc[6380,1:]
			row = np.array(row)
			# print(type(row))

			# padded = np.zeros([1500,130])
			# padded[1500-file.shape[0]:,:file.shape[0]] = file


			data.append(row)
			label.append(1)
			# shapee = np.append(shapee, file.shape[0])
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


			# print(file.shape)
			# file = file.drop(columns=['name', 'frameTime'])
			# print(file.shape)
			# file = file.to_numpy()

			# print('-----------')
			# print(file.shape[0])
			# print()
			row = lines[0:300]
			# row = file.iloc[6380,1:]
			row = np.array(row)
			# print(type(row))

			data.append(row)
			label.append(0)
		# shapee = np.append(shapee, file.shape[0])
		except:
			print("------------ error in file -------- =   {}".format(f[0]))

	data = np.array(data)
	label = np.array(label)

	return data, label






### Train

path_train_S = '/home/helium-balloons/Desktop/midas/C-SIF/train-S.csv'
path_train_NS = '/home/helium-balloons/Desktop/midas/C-SIF/train-NS.csv'

data, label = get_data(path_train_S, path_train_NS)

print(data.shape)
print(label.shape)


### Validation

path_val_S = '/home/helium-balloons/Desktop/midas/C-SIF/val-S.csv'
path_val_NS = '/home/helium-balloons/Desktop/midas/C-SIF/val-NS.csv'

val_data, val_label = get_data(path_val_S, path_val_NS)

### Test

path_test_S = '/home/helium-balloons/Desktop/midas/C-SIF/test-S.csv'
path_test_NS = '/home/helium-balloons/Desktop/midas/C-SIF/test-NS.csv'

test_data, test_label = get_data(path_test_S, path_test_NS)


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


# earlyStopping = EarlyStopping(monitor='val_recall_m', patience=10, verbose=0, mode='max')
# mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_recall_m', mode='max')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_recall_m', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='max')

earlyStopping = EarlyStopping(monitor='val_get_f1', patience=10, verbose=0, mode='max')
mcp_save = ModelCheckpoint('/home/helium-balloons/Desktop/midas/model.mdl_wts.hdf5', save_best_only=True, monitor='val_get_f1', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_get_f1', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='max')


model = tf.keras.Sequential()
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=True), input_shape=(1500, 130)))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, return_sequences=True)))
# model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', input_shape=(300,)))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu', input_shape=(6373,)))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),))
model.add(tf.keras.layers.Activation('sigmoid'))

print(model.summary())

# sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.9, beta_2=0.9, epsilon=1e-07, amsgrad=False,
    name='Adam', clipnorm = 1.0
)

# data = np.concatenate((data, val_data), axis=0)
# label = np.concatenate((label, val_label), axis=0)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy', get_f1, recall_m])

# for i in range(6):
# 	model.fit(data, label, epochs=3, batch_size = 256, validation_data=(test_data,test_label))#, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

# 	print('\n \n \n ')

# 	model.evaluate(test_data,test_label)

# 	print('\n \n \n ')

model.fit(data, label, epochs=200, batch_size = 128, validation_data=(test_data,test_label))#, callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

model.evaluate(test_data, test_label)