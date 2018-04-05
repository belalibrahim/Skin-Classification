import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as k
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop

# For reproducibility
np.random.seed(123)

# Read the data
train = pd.read_csv('Dataset/train_all.csv', index_col='id')
test = pd.read_csv('Dataset/test_all.csv', index_col='id')

# Make the class column contain 0 or 1 only
train['class'] = train['class'] - 1

# Initialize the predictors and the response
predictors = train.drop(['class'], axis=1)
response = to_categorical(train['class'].values, 2)

# Get the number of features
n_cols = predictors.shape[1]

'''
# Build the model
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))

# Stop the training after the loss increases 3 times in a row
early_stopping_monitor = EarlyStopping(patience=3)

# Compile, train and save the model
model.compile(optimizer='adam', loss='categorical_hinge', metrics=['accuracy'])
model.fit(predictors, response, validation_split=0.2, verbose=1, epochs=50, callbacks=[early_stopping_monitor])
model.save('model.h5')
'''

# Load the model
my_model = load_model('model.h5')

# Get the predictions
predictions = np.array(my_model.predict(test), dtype='int')[:, 1]

# Display model summary
my_model.summary()

# Set the data into the suitable format
test = test.drop(['B', 'G', 'R'], axis=1)
test['class'] = (predictions + 1)

# Save the data to csv format
test.to_csv('output.csv', sep=',')
