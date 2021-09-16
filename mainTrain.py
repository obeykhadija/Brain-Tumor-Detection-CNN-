import os

import cv2
import numpy as np
import tensorflow as tf
from keras import utils
from keras.backend import binary_crossentropy
from keras.engine.sequential import Sequential
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical

image_dir = 'datasets/'

no_tumor_img = os.listdir(image_dir+ 'no/')
tumor_img = os.listdir(image_dir+ 'yes/')

data = []
label = []

INPUT_SIZE=64

#Load Images
for i, image_name in enumerate(no_tumor_img):
    if image_name.split('.')[1] == 'jpg':                   #ensure we are looking at jpg images from our folder
        image = cv2.imread(image_dir + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')               #convert to pillow image
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        data.append(np.array(image))                        #add the images to data 
        label.append(0)                                     #add the respective label

for i, image_name in enumerate(tumor_img):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_dir + 'yes/' + image_name) 
        image = Image.fromarray(image, 'RGB')               
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        data.append(np.array(image))
        label.append(1)

#Data Preprocessing
def preprocessing_inputs(data, label):
    #convert to numpy arrays
    X = np.array(data)
    y = np.array(label)

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    #normalize data
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)
    y_train = to_categorical(y_train, num_classes=2)
    y_test= to_categorical(y_test, num_classes=2)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocessing_inputs(data, label)

print('Train set:', X_train.shape, y_train.shape)                                       #(n, image width, image height, n_channels)                            
print('Test set:', X_test.shape, y_test.shape)                                      

#Model Building
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#Binary CrossEntropy = 1, sigmoid
#Categorical CrossEntropy = 2, softmax

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
epochs = 10

#Model Training
model.fit(X_train,y_train, 
          batch_size=32, 
          epochs=epochs, 
          verbose=1, 
          validation_data=(X_test, y_test), 
          shuffle=True)

model_acc = model.evaluate(X_test, y_test)[1]
print('Test accuracy: {:.3%}'.format(model_acc))
