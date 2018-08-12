"""
This is simple a neural network with keras to solve the XOR problem
"""

# import libraries
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


# inputs and labels
XOR_input = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])

XOR_label = np.array([[0],
                      [1],
                      [1],
                      [0]])


# the model
model = keras.Sequential([
    keras.layers.Dense(2), 
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# compile the model
model.compile(optimizer="sgd", # stochastic gradient descent
              loss="mean_squared_error")

# train the model
model.fit(XOR_input, XOR_label, epochs=50000)

# print the predictions
print(model.predict(XOR_input))
