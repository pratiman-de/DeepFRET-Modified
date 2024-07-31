import lib.ml
import lib.utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Bidirectional
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# loading data
data_id = str(np.load("data/data_id.npy"))
X = np.load('data/x_sim_'+data_id+'.npy')
labels = np.load('data/y_sim_'+data_id+'.npy')
set_y = set(labels.ravel())
y = lib.ml.class_to_one_hot(labels, num_classes=len(set_y))
y = lib.ml.smoothe_one_hot_labels(y, amount=0.1)
X = X[..., 0:3]
X = lib.utils.sample_max_normalize_3d(X)
labels = None

def create_lstm_model(n_features, regression):
    inputs = Input(shape=(None, n_features))

    x = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
    final = x

    if regression:
        outputs = TimeDistributed(Dense(1, activation=None))(final)
        metric = "mse"
        loss = "mse"
    else:
        outputs = TimeDistributed(Dense(6, activation="softmax"))(final)
        metric = "accuracy"
        loss = "categorical_crossentropy"

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=[metric])
    return model

def create_gru_model(n_features, regression):
    inputs = Input(shape=(None, n_features))

    x = Bidirectional(GRU(units=128, return_sequences=True))(inputs)
    x = Bidirectional(GRU(units=64, return_sequences=True))(x)
    final = x

    if regression:
        outputs = TimeDistributed(Dense(1, activation=None))(final)
        metric = "mse"
        loss = "mse"
    else:
        outputs = TimeDistributed(Dense(6, activation="softmax"))(final)
        metric = "accuracy"
        loss = "categorical_crossentropy"

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=loss, optimizer="adam", metrics=[metric])
    return model

model = create_lstm_model(n_features=X.shape[-1], regression=False)

'''
if regression:
    X = np.load('data/x_sim_'+data_id+'.npy')
    y = np.load('data/y_sim_'+data_id+'.npy')
'''


lr = 1e-3
minimum_lr = 1e-8
batch_size=24
epochs=5

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                factor=0.5,
                                min_delta=0.05,
                                mode='min',  
                                patience=2,
                                min_lr=minimum_lr,
                                verbose=1)

early_stop = EarlyStopping(monitor="val_loss",  # "val_loss"
                            min_delta=0.05,
                            mode='min',  # on acc has to go max
                            patience=10,
                            verbose=1)

checkpoint = ModelCheckpoint(filepath='output/chk/fret_{epoch:02d}.keras',
                                    monitor='val_loss',
                                    save_weights_only=False,  
                                    save_freq=1)  


history = model.fit(X, y,epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop, reduce_lr, checkpoint])

model.save("output/fret"+data_id+".keras")



date = datetime.now().strftime("%d%H%M")
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
lr = history.history['lr']

file_m = 'output' + "/" + "metrics_" + data_id +".csv"

with open(file_m, 'w') as file:
    file.write('Epoch\tTrain_Loss\tVal_Loss\tTrain_Acc\tVal_Acc\tLR\n')
    min_len = min(len(train_loss), len(val_loss), len(train_acc), len(val_acc), len(lr))
    for epoch in range(min_len):
        file.write(f'{epoch+1}\t{train_loss[epoch]}\t{val_loss[epoch]}\t{train_acc[epoch]}\t{val_acc[epoch]}\t{lr[epoch]}\n')
