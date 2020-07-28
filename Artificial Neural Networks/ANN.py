# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:57:45 2020

@author: SAI AJAY
"""


#import dependencies
import numpy as np
import datetime
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist

#Pre-Processing

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Normalizing Data

X_train = X_train/255.0

X_test = X_test/255.0

#Reshaping

X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

#Buiding a Neural Network
model = tf.keras.models.Sequential()

#Adding 1st layer 
#no_of_output_neurons = 128, activation_func = relu, input_shape = 784
model.add(tf.keras.layers.Dense(units = 128, activation = 'relu', input_shape = (784,)))
#add a dropout layer
model.add(tf.keras.layers.Dropout(0.2))

#Adding Second Layer(output)
model.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))

#compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

#training
model.fit(X_train, y_train, epochs = 5)

#model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)

#saving_model
model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
    
#saving network weights
model.save_weights("fashion_model.h5")




