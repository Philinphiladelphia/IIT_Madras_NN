import keras
from keras.layers import Input ,Dense, Dropout, Activation, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from keras.layers import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import scipy

import numpy as np

from PIL import Image

import pickle


xTrain = pickle.load(open("arrayData2.txt","rb"))
valData = pickle.load(open("valData.txt","rb"))

def makeSlices(xTrain):

    signalLengths = []
    newList = []
    for i in range(len(xTrain)):
        signalLengths.append(len(xTrain[i]))  
    signalLengths.sort()
    for j in range(len(signalLengths)):
        for i in range(len(xTrain)):
            if len(xTrain[i]) == signalLengths[j]:
                newList.append(xTrain[i])
                break
        
    xTrain = newList
    batches = []
    yTrain = []
    for i in range(len(xTrain)):
        yTrain.append(np.empty(len(xTrain)-i))
    #percentage of signal cut away

    for i in range(len(xTrain)): #creates len(xTrain) training batches, each with a sliced signal from each longer signal inside.
        batches.append([])
        for j in range(i,len(xTrain)):
            batches[i].append(xTrain[j][0:signalLengths[i]])
            yTrain[i][j-i] = signalLengths[i] / len(xTrain[j]) #generates percent sliced away for each signal

    return [batches, yTrain]

[batches,yTrain] = makeSlices(xTrain)
[xVal, yVal] = makeSlices(valData)




timesteps=100
number_of_samples=2500
nb_samples=number_of_samples
frame_row=128
frame_col=128
channels=1

nb_epoch=1
batch_size=timesteps

model=Sequential();                          

model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'), input_shape=(None,128,128,1)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Convolution2D(32, 3, 3)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.5)))
#Eliminate Dropout?
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(100))) #decrease size?
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(32, name="first_dense" )))
model.add(TimeDistributed(BatchNormalization()))
#implement stateful?
#model.add(LSTM(20, return_sequences=True, name="lstm_layer"))
model.add(LSTM(20, name="lstm_layer2"))

model.add(Dense(1))



#Change?
#Change pooling function?

#%%

#change solvers?
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mape'])

#graph Loss/MAPE?
#plot %predictions vs real percentages?

histories = []
errors = []
for i in range(len(batches)):
    batchData = batches[i]
    batchData = keras.preprocessing.sequence.pad_sequences(batchData[0:])
    batchData = np.reshape(batchData, [len(batchData),len(batchData[0]),128,128,1])
    batchData = batchData / np.std(batchData)
    
    batch_size = 1

    history = model.fit(batchData, yTrain[i], epochs=3, batch_size=batch_size, verbose = 1)
    histories.append(history.history['mean_absolute_percentage_error'][2]*100)
    score = []
    for j in range(len(xVal)):
        valData = xVal[j]
        valData = keras.preprocessing.sequence.pad_sequences(valData[0:])
        valData = np.reshape(valData, [len(valData),len(valData[0]),128,128,1])
        valData = valData / np.std(valData)
        score.append(model.predict(valData, batch_size=1))
    
    meanError = 0
    for j in range(len(score)):
        meanError = meanError + (abs((np.sum(yVal[j]) - np.sum(score[j]))/np.sum(yVal[j]))/len(yVal[j]))
    meanError = meanError/len(yVal)
    errors.append(meanError)
    i = 1
#Add validation data?
print (errors)
print (histories)
i = 1
#TODO: Shuffle data more thouroughly
#TODO: sort chart by number of epochs/slice size
#TODO increase slice size by several factors to speed things up?
#TODO only use samples more than 50% of the way to fatigue?