import lib.ml
import lib.utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Bidirectional, GRU, TimeDistributed
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from datetime import datetime
import os

# Allow memory growth for GPUs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Loading data
data_id = str(np.load("data/data_id.npy"))
X = np.load('data/x_sim_'+data_id+'.npy')
labels = np.load('data/y_sim_'+data_id+'.npy')
set_y = set(labels.ravel())
y = lib.ml.class_to_one_hot(labels, num_classes=len(set_y))
y = lib.ml.smoothe_one_hot_labels(y, amount=0.1)
X = X[..., 0:3]
X = lib.utils.sample_max_normalize_3d(X)
labels = None

# Function to create LSTM model
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

# Function to create GRU model
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

# Multi-GPU training setup
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_lstm_model(n_features=X.shape[-1], regression=False)

    # Training parameters
    lr = 1e-3
    minimum_lr = 1e-8
    batch_size = 24 * strategy.num_replicas_in_sync  # Scale batch size by the number of GPUs
    epochs = 5

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                  factor=0.5,
                                  min_delta=0.05,
                                  mode='min',
                                  patience=2,
                                  min_lr=minimum_lr,
                                  verbose=1)

    early_stop = EarlyStopping(monitor="val_loss",
                               min_delta=0.05,
                               mode='min',
                               patience=10,
                               verbose=1)

    checkpoint = ModelCheckpoint(filepath='output/chk/fret_{epoch:02d}.keras',
                                 monitor='val_loss',
                                 save_weights_only=False,
                                 save_freq='epoch')

    # Model training
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop, reduce_lr, checkpoint])

    # Save the model
    model.save("output/fret"+data_id+".keras")

# Save training metrics
date = datetime.now().strftime("%d%H%M")
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

file_m = 'output' + "/" + "metrics_" + data_id +".csv"

with open(file_m, 'w') as file:
    file.write('Epoch\tTrain_Loss\tVal_Loss\tTrain_Acc\tVal_Acc\n')
    min_len = min(len(train_loss), len(val_loss), len(train_acc), len(val_acc))
    for epoch in range(min_len):
        file.write(f'{epoch+1}\t{train_loss[epoch]}\t{val_loss[epoch]}\t{train_acc[epoch]}\t{val_acc[epoch]}\n')
