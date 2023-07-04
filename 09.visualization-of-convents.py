#
# Step 1 : Import the neccessary libaries which is important for visualization of contents
#
import os
import pandas as pd
import numpy as np
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import matplotlib.pylab as pylab

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.layers import Conv2D, MaxPool2D
import torch

#
# Step 2 : To stop potential randomness with training and testing data, call the respective data set
#
seed = 128
rng = np.random.RandomState(seed)
data_dir = ''
train = pd.read_csv('')
test = pd.read_csv('')
img_name = rng.choice(train.filename)
filepath = os.path.joi(data_dir, 'train', img_name)
img = imread(filepath, flatten=True)

#
# Step 3 : Plot the necessary images to get the training and testing data defined in perfect way
#
pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()