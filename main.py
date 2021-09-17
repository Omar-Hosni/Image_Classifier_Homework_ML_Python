import tensorflow as tf 
import numpy as np 
import pandas as pd
import keras 
import sklearn
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train.shape
y_train.shape

'''
we had split into training set and test set
now we are going to create validation set
we are going to train the neural network using Gradient Descent
we must scale the input features 
for simplicity, we'll scale the pixel intensity
down to 0-1 range by dividing them by 255.0
also convert them to floats
'''

x_train, x_val = x_train[:55000] /255.0 , x_train[55000:] /255.0
y_train, y_val = y_train[:55000]  , y_train[55000:]

x_test = x_test/255

plt.imshow(x_train[0], cmap='gray')
plt.show()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Building the neural network. Classification of MLP with two hidden layers
#Sequential - simplest kind of keras model for NN, they are composed of single stack of layers connected sequentially

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),     #First layer(Flatten Layer): its role is to convert each input image into a 1D array. if it receives input data X, it computes X.reshape(-1,1). This layer does not have any parameters; it is just there to do some simple processing. Since it is the first layer in the model, you should specify the input_shape, which doesn't include the batch size
    keras.layers.Dense(300, activation="relu"),     #Now we add a Dense hidden layer with 300 neuros, It will use ReLU activation function. Each Dense Layer manages its own weight in matrix, containing all connection weights between neurons and their inputs 
    keras.layers.Dense(100, activation="relu"),     #Then we add a second Dense layer
    keras.layers.Dense(10, activation="softmax")    #Finally we add a Dense output layer with 10 neurons(one per class), using the softmax activation function(because the classes are exclusive)
])

tf.keras.utils.plot_model(model,show_shapes=True)

#Adam optimizer, the optimizer gets the minimum of loss function, gets the weights that reduces the errors, because it has adapter learning rate and momentum, this algorithm is an extension of Gradient Descent
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_test, y_test, epochs=20, validation_data=(x_val, y_val), batch_size=64)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) #gca(get current axis).set_ylim(set y axis limitation)
plt.show()

model.evaluate(x_test, y_test) #evaluating the validation sets
