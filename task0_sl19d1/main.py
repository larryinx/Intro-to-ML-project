import pandas as pd
import numpy as np

# load data
df_train = pd.read_csv('train.csv', header=0, index_col=0)
df_test = pd.read_csv('test.csv', header=0, index_col=0)

# split data
X_train = df_train.drop('y', axis=1)
y_train = df_train['y']
X_test = df_test

# Using tensorflow to build a linear regression model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.RMSprop(0.001),
                metrics=['mean_absolute_error', 'mean_squared_error'])

# Early stopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Train the model
model.fit(X_train, y_train, epochs=1000, verbose=0, callbacks=[early_stop])


# Make predictions and export to .csv
predictions = model.predict(X_test)
predictions_frame = pd.DataFrame(predictions, index=df_test.index, columns=["y"])
predictions_frame.to_csv('sub.csv', sep=",")