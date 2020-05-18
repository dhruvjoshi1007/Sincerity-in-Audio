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
			padded[1500-file.shape[0]:,:file.shape[1]] = file




			row = lines[0:700]
			row = np.array(row)

			data_seq.append(padded)
			data_non_seq.append(row)
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

			file = pd.read_csv('/home/helium-balloons/Desktop/midas/audio_features_egemaps_new/{}{}'.format(f[0].split('.')[0],'.csv'), sep=';')
			file = file.drop(columns=['name', 'frameTime'])
			file = file.to_numpy()

			padded = np.zeros([1500,130])
			padded[1500-file.shape[0]:,:file.shape[1]] = file

			

			row = lines[0:700]

			row = np.array(row)

			data_seq.append(padded)
			data_non_seq.append(row)
			label.append(0)
		except Exception as e:
			print("------------ error in {} -------- =  {}".format(f[0], e))

	data_seq = np.array(data_seq)
	label = np.array(label)
	data_non_seq = np.array(data_non_seq)

	return data_seq, data_non_seq, label



### Train

path_train_S = '/home/helium-balloons/Desktop/midas/C-SIF/train-S.csv'
path_train_NS = '/home/helium-balloons/Desktop/midas/C-SIF/train-NS.csv'

data_seq, data_non_seq, label= get_data(path_train_S, path_train_NS)

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


# earlyStopping = EarlyStopping(monitor='val_recall_m', patience=10, verbose=0, mode='max')
# mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_recall_m', mode='max')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_recall_m', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='max')

earlyStopping = EarlyStopping(monitor='val_recall_m', patience=10, verbose=0, mode='max')
mcp_save = ModelCheckpoint('/home/helium-balloons/Desktop/midas/model.mdl_wts.hdf5', save_best_only=True, monitor='val_recall_m', mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_recall_m', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='max')


# x_seq = tf.placeholder("float", [64, 1500, 130])

# x_non_seq = tf.placeholder("float", [64, 300])

# y_true = tf.placeholder("float", [64, 1])

# bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, 
# 							# kernel_regularizer=tf.keras.regularizers.l2(0.01),
# 							return_sequences=True))(x_seq)

inputA = tf.keras.layers.Input([1500,130])
inputB = tf.keras.layers.Input([700])

bilstm = tf.keras.layers.LSTM(32,#dropout=0.5,
							# kernel_regularizer=tf.keras.regularizers.l2(0.01),
							return_sequences=True)(inputA)

flat = tf.keras.layers.Flatten()(bilstm)

dense1 = tf.keras.layers.Dense(64, activation='relu')(flat)

mix = tf.concat([dense1,inputB], axis=1)

dense2 = tf.keras.layers.Dense(64, activation='relu')(mix)

output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

model = tf.keras.models.Model(inputs=[inputA, inputB], outputs=output)


sgd = tf.keras.optimizers.SGD(lr=0.01, decay=0.01/10, momentum=0.7, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy', get_f1])

model.fit([data_seq, data_non_seq], label, epochs=15, validation_data=[[val_data_seq, val_data_non_seq], val_label])

model.evaluate([val_data_seq, val_data_non_seq], val_label)

model.fit([data_seq, data_non_seq], label, epochs=15, validation_data=[[val_data_seq, val_data_non_seq], val_label])

model.evaluate([val_data_seq, val_data_non_seq], val_label)

# bce = tf.keras.losses.BinaryCrossentropy()

# loss = bce(y_true, output)

# variables_names = [v.name for v in tf.trainable_variables()]

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9).minimize(loss, tf.compat.v1.trainable_variables())


# def batch_generator(x1, x2, y, batch_size = 64):
# 	indices = np.arange(x1.shape[0]) 
# 	batch_x1 = np.array([])
# 	batch_x2 = np.array([])
# 	batch_y = np.array([])
# 	while True:
# 			# it might be a good idea to shuffle your data before each epoch
# 		np.random.shuffle(indices) 
# 		for i in indices:
# 			batch_x1 = np.append(batch_x1,x1[i])
# 			batch_x2 = np.append(batch_x2,x2[i])
# 			batch_y = np.append(batch_y, y[i])

# 			if batch_y.shape[0]==batch_size:
# 				break
# 		return batch_x1, batch_x2, batch_y

# sess = tf.Session()
# with sess.as_default():
# 	sess.run(tf.global_variables_initializer())

# 	for i in range(20):

# 		x_train_seq, x_train_non_seq, y_train = batch_generator(data_seq, data_non_seq, label, 64)
# 		loss_val, out, _ = sess.run([loss, output, optimizer], feed_dict={x_seq:x_train_seq, x_train_non_seq: data_non_seq, y_true:y_train})

# 		print(loss_val)
