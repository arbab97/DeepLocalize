import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tensorflow.python.keras.datasets import mnist
from model.loss import *
from model.layers import *
from model.network import *
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
np.random.seed(1)
# ! Warning: All warnings are ignored. 

custom_data_path='/media/rabi/Data/11111/Task 99/deepcrime/datasets/custom_data.csv'
data = pd.read_csv(custom_data_path)
# data.head()
# data.isnull().sum()

data,test = train_test_split(data,test_size = 0.2)
#data,validation = train_test_split(data,test_size = 0.2)
# data.shape

data = data.drop(columns = ['id'])
#data['diagnosis'].unique()   #comment this

labels = data['diagnosis']
data.drop(columns = ['diagnosis'],inplace = True)
data = data.iloc[:,0:29]
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

#To adjust the format: 
training_data = data_x.reshape(455, 1, 1, 29)
training_labels = np.array(labels).reshape(455,1)

lr = 0.01
checker = False
net = Sequential()
net.add(Flatten())
net.add(Dense(num_inputs=29, num_outputs=12, learning_rate=lr, name='fc1'))
net.add(ReLu())
net.add(Dropout(0.5))
net.add(Dense(num_inputs=12, num_outputs=5, learning_rate=lr, name='fc2'))
net.add(ReLu())
net.add(Dropout(0.5))
net.add(Dense(num_inputs=5, num_outputs=1, learning_rate=lr, name='fc3'))
net.add(Tanh())
print('Training Custom Network......')
net.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
net.fit(training_data, training_labels, 32, 2) #batch size 1 and 5 epochs
print("Done")

#___________________________________________________________________________________________________
# #Base Model
# sys.exit()
# model = Sequential()
# model.add( Dense(12,input_dim =29,activation = 'relu'))  #Warning: The mnist one uses input_shape !
# model.add(Dropout(0.5))#
# model.add(Dense(5,activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1,activation = 'softmax'))
# model.compile(loss = keras.losses.binary_crossentropy,optimizer =keras.optimizers.Adam(),metrics = ['accuracy'])
# # New Mapping: x_train->data_x; y_train->labels; x_test-> test; y_test->test_y
# model.fit(data_x, labels, batch_size=batch_size, epochs=epochs, verbose=1)
 
#  model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_1 (Dense)              (None, 12)                360       
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 12)                0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 5)                 65        
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 5)                 0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 6         
# =================================================================
