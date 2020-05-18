import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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


# filename = '/home/helium-balloons/Desktop/midas/audio/806_P-30_f_S-6_PS-M.wav'

# y, sr = librosa.load(filename, duration=7)

# mfcc = librosa.feature.mfcc(y=y, sr=sr).T

# chroma = librosa.feature.chroma_stft(y=y, sr=sr).T

# mel = librosa.feature.melspectrogram(y=y, sr=sr).T

# result=np.hstack((mfcc, chroma, mel))

# print(y.T.shape)
# print(sr)
# print(mfcc.shape)
# print(chroma.shape)
# print(mel.shape)
# print(result.shape)


# sample_rate = 22050
sample_rate = 100


def get_data(path_S, path_NS):

    filenames_S = pd.read_csv(path_S)
    filenames_NS = pd.read_csv(path_NS)
    filenames_S = filenames_S.to_numpy()
    filenames_NS = filenames_NS.to_numpy()
    print(filenames_S.shape)
    # print(filenames_S)

    data_seq = []
    data_non_seq = []
    data_ext = []
    label = []
    shapee = np.array([])

    ctr = 0

    for f in filenames_S:
        # print(f)
        if ctr % 10 == 0:
            print('loading S data {}'.format(ctr))
        ctr = ctr + 1
        try:
            # file = pd.read_csv('/home/helium-balloons/Desktop/midas/audio_file/{}{}'.format(f[0],'.csv'))
            data_file = '/home/helium-balloons/Desktop/midas/audio_file/{}{}'.format(f[0],'.csv')
            with open(data_file, 'r') as temp_f:
            # Read the lines
                lines = temp_f.readlines()
                lines = lines[-1].split(',')
                lines = [float(x) for x in lines[1:len(lines)-1]]

            file = pd.read_csv('/home/helium-balloons/Desktop/midas/audio_features_egemaps_2/{}{}'.format(f[0].split('.')[0],'.csv'), sep=';')

            file = file.drop(columns=['name', 'frameTime'])
            file = file.to_numpy()


            padded = np.zeros([200,130])
            padded[200-file.shape[0]:,:file.shape[1]] = file

            
            y, sr = librosa.load('/home/helium-balloons/Desktop/midas/audio/{}'.format(f[0]), duration=7, sr=sample_rate)

            mfcc = librosa.feature.mfcc(y=y, sr=sr).T

            chroma = librosa.feature.chroma_stft(y=y, sr=sr).T

            mel = librosa.feature.melspectrogram(y=y, sr=sr).T

            result=np.hstack((mfcc, chroma, mel))

            padded1 = np.zeros([310,160])
            padded1[310-result.shape[0]:,:result.shape[1]] = result


            row = lines[0:300]
            row = np.array(row)

            data_seq.append(padded)
            data_non_seq.append(row)
            data_ext.append(np.array(padded1))
            label.append(1)
            
        except Exception as e:
            print("------------ error in {} -------- =  {}".format(f[0], e))

    # shapee  = np.array([])

    for f in filenames_NS:

        if ctr % 10 == 0:
            print('loading NS data {}'.format(ctr))

        try:
            data_file = '/home/helium-balloons/Desktop/midas/audio_file/{}{}'.format(f[0],'.csv')
            with open(data_file, 'r') as temp_f:
            # Read the lines
                lines = temp_f.readlines()
                lines = lines[-1].split(',')
                    # print(lines)
                lines = [float(x) for x in lines[1:len(lines)-1]]

            file = pd.read_csv('/home/helium-balloons/Desktop/midas/audio_features_egemaps_2/{}{}'.format(f[0].split('.')[0],'.csv'), sep=';')
            file = file.drop(columns=['name', 'frameTime'])
            file = file.to_numpy()

            padded = np.zeros([200,130])
            padded[200-file.shape[0]:,:file.shape[1]] = file

            y, sr = librosa.load('/home/helium-balloons/Desktop/midas/audio/{}'.format(f[0]), duration=7, sr=sample_rate)

            mfcc = librosa.feature.mfcc(y=y, sr=sr).T

            chroma = librosa.feature.chroma_stft(y=y, sr=sr).T

            mel = librosa.feature.melspectrogram(y=y, sr=sr).T

            result=np.hstack((mfcc, chroma, mel))

            padded1 = np.zeros([310,160])
            padded1[310-result.shape[0]:,:result.shape[1]] = result

            

            row = lines[0:300]

            row = np.array(row)

            data_seq.append(padded)
            data_non_seq.append(row)
            data_ext.append(np.array(padded1))
            label.append(0)
        except Exception as e:
            print("------------ error in {} -------- =  {}".format(f[0], e))

    data_seq = np.array(data_seq)
    data_ext = np.array(data_ext)
    label = np.array(label)
    data_non_seq = np.array(data_non_seq)

    return data_seq, data_non_seq, data_ext, label




path_train_S = '/home/helium-balloons/Desktop/midas/C-SIF/train-S.csv'
path_train_NS = '/home/helium-balloons/Desktop/midas/C-SIF/train-NS.csv'

data_seq, data_non_seq, data_ext, label= get_data(path_train_S, path_train_NS)

# print(data.shape)
# print(label.shape)


### Validation

path_val_S = '/home/helium-balloons/Desktop/midas/C-SIF/val-S.csv'
path_val_NS = '/home/helium-balloons/Desktop/midas/C-SIF/val-NS.csv'

val_data_seq, val_data_non_seq, val_data_ext, val_label = get_data(path_val_S, path_val_NS)

### Test

path_test_S = '/home/helium-balloons/Desktop/midas/C-SIF/test-S.csv'
path_test_NS = '/home/helium-balloons/Desktop/midas/C-SIF/test-NS.csv'

test_data_seq, test_data_non_seq, test_data_ext, test_label = get_data(path_test_S, path_test_NS)


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
#                           # kernel_regularizer=tf.keras.regularizers.l2(0.01),
#                           return_sequences=True))(x_seq)

inputA = tf.keras.layers.Input([200,130])
inputB = tf.keras.layers.Input([300])
inputC = tf.keras.layers.Input([310, 160])

bilstm = tf.keras.layers.LSTM(16,#dropout=0.5,
                            # kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            return_sequences=True)(inputA)

flat = tf.keras.layers.Flatten()(bilstm)

bilstm2 = tf.keras.layers.LSTM(16,#dropout=0.5,
                            # kernel_regularizer=tf.keras.regularizers.l2(0.01),
                            return_sequences=True)(inputC)

flat2 = tf.keras.layers.Flatten()(bilstm2)

dense0 = tf.keras.layers.Dense(64, activation='relu')(flat2)

drop0 = tf.keras.layers.Dropout(0.2)(dense0)

dense1 = tf.keras.layers.Dense(64, activation='relu')(flat)

drop1 = tf.keras.layers.Dropout(0.2)(dense1)

mix = tf.concat([drop1,inputB, drop0], axis=1)

dense2 = tf.keras.layers.Dense(64, activation='relu')(mix)

drop2 = tf.keras.layers.Dropout(0.2)(dense2)

output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.models.Model(inputs=[inputA, inputB, inputC], outputs=output)


sgd = tf.keras.optimizers.SGD(lr=0.01, decay=0.01/10, momentum=0.7, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', get_f1])

model.fit([data_seq, data_non_seq, data_ext], label, epochs=15, validation_data=[[val_data_seq, val_data_non_seq, val_data_ext], val_label])

model.evaluate([val_data_seq, val_data_non_seq, val_data_ext], val_label)

model.fit([data_seq, data_non_seq, data_ext], label, epochs=15, validation_data=[[val_data_seq, val_data_non_seq, val_data_ext], val_label])

model.evaluate([val_data_seq, val_data_non_seq, val_data_ext], val_label)
