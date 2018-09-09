import keras
from keras.layers import Input ,Dense, Dropout, Activation, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import scipy

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

import pickle

def normalize(myList):
    normList = []
    for i in range(len(myList)):
        normList.append(myList[i]-min(myList))/(max(myList)-min(myList))
    
    return normList
##################### PARAMETERS
dropout = 0.1


######################

xTrain = pickle.load(open("arrayData2.txt","rb"))
valData = pickle.load(open("valData.txt","rb"))

min = 100000
valMin = 100000


for i in range(len(xTrain)):
    if np.shape(xTrain[i])[0] < min:
        min = np.shape(xTrain[i])[0]
    i = np.shape(xTrain[i])[1]

for i in range(len(valData)):
    if np.shape(valData[i])[0] < valMin:
        valMin = np.shape(valData[i])[0]
    i = np.shape(valData[i])[1]

yTrain = np.empty(len(xTrain))
yVal = np.empty(len(valData))

for i in range(len(xTrain)):
    yTrain[i] = min / len(xTrain[i])  #percentage of signal cut away
    xTrain[i] = xTrain[i][0:min]

for i in range(len(valData)):
    if len(valData[i]) > min:
        yVal[i] = min / len(valData[i])  #percentage of signal cut away
        valData[i] = valData[i][0:min]
    else:
        yVal[i] = 1.0



######## OLD BLOCK ##############
xTrain = keras.preprocessing.sequence.pad_sequences(xTrain[0:])
valData = keras.preprocessing.sequence.pad_sequences(valData[0:],min) #might be borked, check
#for i in range(len(xTrain)):
   # xTrain[i] = np.array(xTrain[i])
######## OLD BLOCK ##############

xTrain = np.reshape(xTrain, [len(xTrain),len(xTrain[1]),128,128,1])
valData = np.reshape(valData, [len(valData),len(valData[1]),128,128,1])

#xTrain = np.concatenate(xTrain[0:],1)
#yTrain = np.empty([5,len(xTrain[1]),1])




timesteps=100
number_of_samples=2500
nb_samples=number_of_samples
frame_row=128
frame_col=128
channels=1

nb_epoch=1
batch_size=timesteps



#data= np.random.random((2500,timesteps,frame_row,frame_col,channels))
#label=np.random.random((2500,1))

#X_train=data[0:2000,:]
#y_train=label[0:2000]

#X_test=data[2000:,:]
#y_test=label[2000:,:]

#%%


#Lengthen Network
model=Sequential();                          

model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'), input_shape=(min,128,128,1)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Convolution2D(32, 3, 3)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(dropout)))
#Eliminate Dropout?
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512)))
                
                
model.add(TimeDistributed(Dense(32, name="first_dense" )))
        

#implement stateful?
model.add(LSTM(20, return_sequences=True, name="lstm_layer"))

model.add(TimeDistributed(Dense(1), name="time_distr_dense_one"))
model.add(GlobalAveragePooling1D(name="global_avg"))
#Change pooling function?

#%%

#change solvers?
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mape'])

#graph Loss/MAPE?
#plot %predictions vs real percentages?
history = model.fit(xTrain, yTrain, epochs=10, batch_size=1, verbose = 1) #turn up batch size?
#Add validation data?
score = model.evaluate(valData, yVal, batch_size=1)

plt.plot(history.history['loss'])
plt.show()
plt.plot(history.history['mean_absolute_percentage_error'])
plt.show()
plt.plot(score)
plt.show()

i = 1

