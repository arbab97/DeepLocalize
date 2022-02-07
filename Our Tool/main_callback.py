
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import keras, sys
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
# ! Warning: All warnings are ignored. 
np.random.seed(1)

custom_data_path='/media/rabi/Data/11111/Task 99/deepcrime/datasets/custom_data.csv'
data = pd.read_csv(custom_data_path)

data,test = train_test_split(data,test_size = 0.2)

data = data.drop(columns = ['id'])

labels = data['diagnosis']
data.drop(columns = ['diagnosis'],inplace = True)
data = data.iloc[:,0:29]
#data.head() 

data_x = data

n = Normalizer()
data_x = n.fit_transform(data_x)

map = {'M':1,'B':0}

labels = labels.map(map)
# labels

test_y = test['diagnosis']
test = test.drop(columns = ['id','diagnosis'])
test= test.iloc[:,0:29]

test = n.transform(test)
test_y = test_y.map(map)
test_y.head()

print("here")
#  Mapping: x_train->data_x; y_train->labels; x_test-> test; y_test->test_y

# learn rate should be 0.01
epochs = 2   # Initial was 600
batch_size = 32 # Default of mnist was 128
# Foramt: callback = keras.callbacks.DeepLocalize(inputs, outputs, layer_number, batch_size, startTime)
callback = keras.callbacks.DeepLocalize(data_x, labels, 5, batch_size, 0)

model = Sequential()
model.add( Dense(12,input_dim =29,activation = 'relu'))  #Warning: The mnist one uses input_shape !
model.add(Dropout(0.5))
model.add(Dense(5,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'exponential'))

model.compile(loss = keras.losses.binary_crossentropy,optimizer =keras.optimizers.Adam(lr=0.001),metrics = ['accuracy'])
# New Mapping: x_train->data_x; y_train->labels; x_test-> test; y_test->test_y
model.fit(data_x, labels, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[callback])

# model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10, batch_size=1, 
#                 callbacks=[callback], verbose=0)

#######!!!batch size ==1 ?
print("End")


# #Simple Executable Example
# callback = keras.callbacks.DeepLocalize(np.arange(100).reshape(5, 20), np.zeros(5), 3, 1, 0)
# model = keras.models.Sequential()
# model.add(keras.layers.Dense(64))
# model.add(keras.layers.Activation(activations.relu))
# model.add(keras.layers.Dense(1))
# model.compile(keras.optimizers.SGD(), loss='mse')
# model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10, batch_size=1, 
#                 callbacks=[callback], verbose=0)