#
# Step 1 : Import the necessary modules for performance of sequence processing with convents
#
import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

#
# Step 2 : Perform the necessary operations to create a pattern in respective sequence
#
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#
# Step 3 : Compile the model and fit the pattern in the mentioned conventional neural network model
#
model.compile(loss = keras.losses.categorical_crossentropy, 
optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test)) 
score = model.evaluate(x_test, y_test, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])