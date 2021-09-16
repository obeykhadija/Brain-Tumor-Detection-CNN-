# Brain-Tumor-Detection-CNN
## Brain Tumor Classification Using Deep Learning

## Getting Started

This is the repository contains the datasets and python files for my brain tumor detection app. I have used a cconvaluted neural network (CNN) to classify MRI Brain Images as either having a tumor or no tumor.

The dataset was collected from [Kaggle](https://www.kaggle.com/ahmedhamada0/brain-tumor-detection). The dataset contains 3000 images in total. Half of the images had no tumor present while the other half had tumor. This dataset was extremly useful since the data was perfectly balanced. Lastly, 60 files were provided in the "pred" folder which serves as the test set.

## Running the App

Running the app is simple but requires you to download some libraries which can be easily downloaded via pip install on your command line:
* import os
* import numpy
* import tensorflow as tf
* import keras import utils
* import sklearn
* import PIL
* import flask
* import werkzeug

Once libraries are downloaded simply run the app in your prefered IDE. Next, you should see a link to http://127.0.0.1:5000/ which you can open on your perferred browser. From here simply choose a file from the pred folder and click predict. 
If you simply wish to see the CNN work without opening the web app you can use the mainTest.py file. Here you can load your selected image from OpenCV through the 
``cv2.imread("pred\'imagename here'")`` function and simpy run the file in terminal. 0 = no tumor, 1 = tumor.

## Results
![results](https://user-images.githubusercontent.com/81998785/133630499-debfaa7a-17b9-442e-9c27-59b4b103a968.JPG)
## Demo
![bandicam 2021-09-16 17-30-40-465](https://user-images.githubusercontent.com/81998785/133632175-27bb7708-2781-4cf8-862c-46ae2cb15412.gif)

