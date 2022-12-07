import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.metrics as metrics

import keras.backend as K


'''
y = (m, max_length, 3)
'''
# def my_loss(y_true, y_output):
    
#     true = K.reshape(y_true, (-1, y_true.shape[-1]))
#     output = K.reshape(y_output, (-1, y_output.shape[-1]))
    
#     bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#     mse = tf.keras.losses.MeanSquaredError()
    
#     # get mse of only true positives
#     true_sack_mask = y_true[:1]==1
#     # checked, mse function can take in array of length zero
#     b = bce(true[:,0:-1], output[:,0:-1])
#     m = mse(true[true_sack_mask][:,-1], output[true_sack_mask][:,-1])
#     return b + m

def my_loss(y_true, y_output):
    
    true = K.reshape(y_true, (-1, y_true.shape[-1]))
    output = K.reshape(y_output, (-1, y_output.shape[-1]))
    
    bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    mse = tf.keras.losses.MeanSquaredError()
 
    return bce(true[:,0:-1], output[:,0:-1]) + mse(true[:,-1], output[:,-1])

def bce_metric(y_true, y_output):
    true = K.reshape(y_true, (-1, y_true.shape[-1]))
    output = K.reshape(y_output, (-1, y_output.shape[-1]))
    return K.mean(K.binary_crossentropy(true[:,0:-1], output[:,0:-1], from_logits=False))

def mse_metric(y_true, y_output):
    return K.mean(K.square(y_output[:,-1] - y_true[:,-1]), axis=-1)

# def mse_metric(y_true, y_output):
#     true = K.reshape(y_true, (-1, y_true.shape[-1]))
#     output = K.reshape(y_output, (-1, y_output.shape[-1]))
#     # get mse of only true positives
#     true_sack_mask = true[:1]==1
#     if len(y_true[true_sack_mask]) == 0:
#         return 0.0
#     else:
#         return K.mean(K.square(true[true_sack_mask][:,-1] - output[true_sack_mask][:,-1]), axis=-1)
    
def accuracy_metric(y_true, y_output):
    true = K.reshape(y_true, (-1, y_true.shape[-1]))
    output = K.reshape(y_output, (-1, y_output.shape[-1]))
    preds = K.cast(K.argmax(output[:,0:-1], axis=-1), 'float32')
    return K.mean(K.cast(true[:,1] == preds, 'float32'))

# https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def recall(y_true, y_output):
    true = K.reshape(y_true, (-1, y_true.shape[-1]))
    output = K.reshape(y_output, (-1, y_output.shape[-1]))
    
    preds = K.cast(K.argmax(output[:,0:-1], axis=-1), 'float32')
    true_positives = K.sum(K.round(K.clip(true[:,1] * preds, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(true[:,1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_output):
    true = K.reshape(y_true, (-1, y_true.shape[-1]))
    output = K.reshape(y_output, (-1, y_output.shape[-1]))
    
    preds = K.cast(K.argmax(output[:,0:-1], axis=-1), 'float32')
    true_positives = K.sum(K.round(K.clip(true[:,1] * preds, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(preds, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def createModel(input_shape = (203, 23*7)):
    
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

x_train = np.load("./seq_unnorm_data/x_train.npy")
y_train = np.load("./seq_unnorm_data/y_train.npy")
x_val = np.load("./seq_unnorm_data/x_val.npy")
y_val= np.load("./seq_unnorm_data/y_val.npy")

MAX_PLAY_LENGTH = 203

model = createModel()

LEARNING_RATE = 0.000001
BETA_1 = 0.9
BETA_2 = 0.999
EPS = 1e-07

opt = optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=BETA_1,
    beta_2=BETA_2,
    epsilon=EPS,
    clipvalue=0.1)

# Better optimizer
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=10000,
    decay_rate=0.9)

scheduled_opt = optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=BETA_1,
    beta_2=BETA_2,
    epsilon=EPS)

model.compile(loss = my_loss, optimizer = scheduled_opt, metrics = [accuracy_metric, bce_metric, mse_metric, recall, precision])
print(f"model compiled")

x_train_input = x_train.reshape(-1, MAX_PLAY_LENGTH, 23, 11)[:,:,:,4:].reshape(-1,MAX_PLAY_LENGTH,23*7)

print(f"input (X) shape = {x_train_input.shape}")
print(f"y shape = {y_train.shape}")

NUM_EPOCHS = 10
BATCH_SIZE = 32
history = model.fit(x_train_input, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))
print(f"model done training")

model_string = f"./rnn_model_unnorm/weights/weights_epochs{NUM_EPOCHS}"
model.save_weights(model_string)

metrics_df = pd.DataFrame(history.history)
metrics_df_csv_string = f"./rnn_model_unnorm/stats/training_metrics"
metrics_df.to_csv(metrics_df_csv_string)

x_val_input = x_val.reshape(-1, MAX_PLAY_LENGTH, 23, 11)[:,:,:,4:].reshape(-1,MAX_PLAY_LENGTH,23*7)
val_loss, val_accuracy, val_bce, val_mse, val_recall, val_precision = model.evaluate(x_val_input, y_val, verbose=2)

val_df = pd.DataFrame([[val_loss, val_accuracy, val_bce, val_mse, val_recall, val_precision]], columns=['val_loss', 'cat_acc', 'val_bce', 'val_mse', 'val_recall', 'val_precision'])
val_df_csv_string = f"./rnn_model_unnorm/stats/val_metrics"
val_df.to_csv(val_df_csv_string)

print(f"val loss = {val_loss}")
print(f"val accuracy = {val_accuracy}")
print(f"val_bce = {val_bce}")
print(f"val_mse = {val_mse}")
print(f"val_recall = {val_recall}")
print(f"val_precision = {val_precision}")
