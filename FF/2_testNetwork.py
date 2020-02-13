import random as rdm
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt        

# FOR TESTING

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# since we're only interested in testing the network.
# only normalize the x_test series
x_test = tf.keras.utils.normalize(x_test, axis=1)

#same logic as before (convert everything to B&W)
for test in range(len(x_test)):
    for row in range(28):
        for x in range(28):
            # if x_test[test][row][x] != 0:
            #    x_test[test][row][x] = 1
            if x_test[test][row][x] > 0.3:
                x_test[test][row][x] = 1
            else:
                x_test[test][row][x] = 0


# lets load a model vs anything else
model = tf.keras.models.load_model('solomon.model')

# Lets make some predictions
print(len(x_test))

# the first ten items in the array
numbers_to_predict = x_test[:10]
predictions = model.predict(np.stack(numbers_to_predict, axis=0))

count = 0
for x in range(len(predictions)):
    # we take the predictions max value across all its values
    guess = (np.argmax(predictions[x]))
    # y_test contains the label or the truth 
    # of what the actual value of the number realy is.
    actual = y_test[x]
    print("I predict this number is a:", guess)
    print("Number Actually Is a:", actual)
    if guess != actual:
        #print("--------------")
        #print('WRONG')
        #print('---------------')
        count+=1
    plt.imshow(x_test[x], cmap=plt.cm.binary)
    plt.show()

print("The program got", count, 'wrong, out of', len(x_test))
print(str(100 - ((count/len(predictions))*100)) + '% correct')

