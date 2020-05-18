import numpy as np
import pandas as pd

filenames = pd.read_csv('/home/helium-balloons/Desktop/midas/sincerity-metadata.csv', sep=';')

data = filenames.to_numpy()

data_1 = []
data_2 = []
data_3 = []
data_4 = []
data_5 = []
data_6 = []


for i in range(data.shape[0]):

	if(data[i][0].split(',')[4] == 'S-1'):
		data_1.append(np.array(data[i][0].split(',')))

	if(data[i][0].split(',')[4] == 'S-2'):
		data_2.append(np.array(data[i][0].split(',')))

	if(data[i][0].split(',')[4] == 'S-3'):
		data_3.append(np.array(data[i][0].split(',')))

	if(data[i][0].split(',')[4] == 'S-4'):
		data_4.append(np.array(data[i][0].split(',')))

	if(data[i][0].split(',')[4] == 'S-5'):
		data_5.append(np.array(data[i][0].split(',')))

	if(data[i][0].split(',')[4] == 'S-6'):
		data_6.append(np.array(data[i][0].split(',')))

data_1 = np.array(data_1)
data_2 = np.array(data_2)
data_3 = np.array(data_3)
data_4 = np.array(data_4)
data_5 = np.array(data_5)
data_6 = np.array(data_6)


print(data_1.shape)
print(data_2.shape)
print(data_3.shape)
print(data_4.shape)
print(data_5.shape)
print(data_6.shape)


def second_split(arr):

	


def first_split(arr):

	data_s = []
	data_ns = []

	for i in range(arr.shape[0]):

		if(arr[i][6] == 'S'):

			data_s.append(arr[i])
		else:
			data_ns.append(arr[i])

	data_s = np.array(data_s)
	data_ns = np.array(data_ns)


	print(data_s.shape)
	print(data_ns.shape)


next_split(data_6)