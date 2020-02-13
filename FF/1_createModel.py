# Used to train and save a neural network on the mnist dataset

import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#specifying dataset 
mnist = tf.keras.datasets.mnist

#splitting the data set into training set and test set
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# place the values between 0 and 1 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# loop over the traning data
# sets it equal to 1 if not zero. 
# this makes it only binary (on or off) which i guess is fine for a computer screen input
# since when you draw the digit its only black or white. 
# the MNIST dataset does have a gradient scale to the image 
# it might be better to make some of these values white/0 (background) vs black/1 (the pen)
for train in range(len(x_train)):
    # images are 28x28 pixels
    for row in range(28):
        for x in range(28):
            if x_train[train][row][x] != 0:
               x_train[train][row][x] = 1

# build a Sequential "Deep" Neural Net
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model train
model.fit(x_train, y_train, epochs=43

# save the trained model
model.save('solomon.model')

print("Model saved")
