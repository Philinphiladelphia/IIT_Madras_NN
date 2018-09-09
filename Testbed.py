import scipy.io as sio
from scipy import signal
from scipy import array
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import png
import PIL
from PIL import Image
import scipy.misc

with open("signals.txt","rb") as fp:
    a= pickle.load(fp)

for i in range(len(a)):
    for j in range(int(np.floor(len(a[i])/1000))):
        cwtData = signal.cwt(array(a[i][1+(1000*j):(j+1)*1000]),signal.ricker,np.arange(1, 31))
        
        #fig, ax = plt.subplots()
        #ax.plot(range(10))
        #fig.patch.set_visible(False)
        #ax.axis('off')
        #plt.imshow(cwtData, extent=[-1, 1, 31, 1], cmap='gray', aspect='auto',vmax=abs(cwtData).max(), vmin=-abs(cwtData).max())

        im = Image.fromarray(cwtData)
        im = im.resize((128,128))
        scipy.misc.imsave('/Users/philjones/Desktop/ConvNet3 Test/waveletImgs/'+str(i+1)+'/'+str(j)+'Wave.jpg', im)
    q = 1
