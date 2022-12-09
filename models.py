import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model


def createModel(input_shape = (23,7)):
    
    X = tfl.Input(input_shape)  # define the input to the model
    flat = tfl.Flatten(input_shape=(23, 7))(X)     # Flatten to pass into linear layers
    d1 = tfl.Dense(50, activation='relu')(flat)
    d3 = tfl.Dense(3,activation=None)(d1)
    
    # have layer (batch_size, 3). Want to take (b, [0,1]) and turn them into probabilities, and keep (b, [2]) as time
    # https://datascience.stackexchange.com/questions/86740/how-to-slice-an-input-in-keras
    intermediate = tfl.Reshape((3,1), input_shape=(3,))(d3)
    
    probs = tfl.Cropping1D(cropping=(0,1))(intermediate)
    probs = tfl.Reshape((2,), input_shape=(2,1))(probs)
    probs = tfl.Activation('softmax')(probs)
    
    time = tfl.Cropping1D(cropping=(2,0))(intermediate)
    time = tfl.Reshape((1,), input_shape=(1,1))(time)
    
    # concatenate the probabilities and predicted_time_to_sack back into one layer
    out = tfl.Concatenate(axis=-1)([probs, time])
    
    model = Model(inputs=X, outputs=out)        # create model
    
    return model

''' 
LSTM architecture
'''
def createLSTMModel(input_shape = (203, 23*7)):
    
    X = tfl.Input(input_shape)  # define the input to the model
    lstm = tfl.LSTM(100, activation='tanh', recurrent_activation='tanh', return_sequences=True)(X)
    #drop = tfl.Dropout(0.2)(lstm)
    d2 = tfl.Dense(50,activation=None)(lstm)
    d3 = tfl.Dense(3,activation=None)(d2)
    permute = tfl.Permute((2,1))(d3)    # change input from (None, 203, 3) to (None, 3, 203)
    
    # have layer (batch_size, 3). Want to take (b, [0,1]) and turn them into probabilities, and keep (b, [2]) as time
    # https://datascience.stackexchange.com/questions/86740/how-to-slice-an-input-in-keras
    probs = tfl.Cropping1D(cropping=(0,1))(permute) # shape (None, 2, 203)
    probs = tfl.Softmax(axis=1)(probs)
    
    time = tfl.Cropping1D(cropping=(2,0))(permute) # shape (None, 1, 203)
    
    # concatenate the probabilities and predicted_time_to_sack back into one layer
    out = tfl.Concatenate(axis=1)([probs, time]) # shape (None, 3, 203)
    out = tfl.Permute((2,1))(out) # shape (None, 203, 3)
    
    model = Model(inputs=X, outputs=out)        # create model
    
    return model