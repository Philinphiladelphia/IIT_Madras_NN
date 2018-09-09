import PIL
from PIL import Image
import numpy as np
from numpy import random
from keras.preprocessing.image import img_to_array
import argparse
from os import listdir
import pickle

xTrain = []

for i in range(len(listdir('/Users/philjones/Desktop/waveletImgs'))-1):
    xTrain.append([])

    for j in range(len(listdir('/Users/philjones/Desktop/waveletImgs/'+str(i+1)))-1):
        im = Image.open('/Users/philjones/Desktop/waveletImgs/'+str(i+1)+'/'+str(j)+'Wave.jpg')
        imArray = img_to_array(im) 
        xTrain[i].append(imArray)

pickle.dump( xTrain, open("arrayData2.txt","wb"))

random.seed(42)
random.shuffle(imagePaths)