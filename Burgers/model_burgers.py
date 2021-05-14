import pathlib
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model_name', help='Name of model', type=str)
parser.add_argument('--N', help='Number of points before/after in domain -- Must be identical to training data', default=3, type=int)
parser.add_argument('--alpha', help='Optimizer learning rate', default=0.001, type=float)
parser.add_argument('-p', '--plot', help='Plot training/validation loss', action="store_true")
args = parser.parse_args()

N = args.N #2N+1 points used to train
Model_Name = args.model_name #Name model to save
OptVal = args.alpha #Learning rate
PLOT = args.plot #Plot

dataset_path = "Data/N%i_data.txt"%N #Location of training data
column_names_1 = ['nu'] #First column name
column_names_2 = ['u%i'%i for i in range(1,(2*N+1)+1)] #Name last columns
column_names = column_names_1 + column_names_2 #Create list of column names
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                            na_values = "?", comment='\t',
                                                  sep=" ", skipinitialspace=True) #Create df
dataset = raw_dataset.copy()
dataset = dataset.dropna() #Drop nans
train_dataset = dataset.sample(frac=0.8,random_state=0) #80% of dataset is for training
test_dataset = dataset.drop(train_dataset.index) #Remaining 20% used for validation
train_stats = train_dataset.describe()
train_stats.pop("nu") #Drop nu vals (labels)
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('nu') #Remove nu column from train labels
test_labels = test_dataset.pop('nu') #Remove nu column from test labels

normed_train_data = train_dataset
normed_test_data = test_dataset

def build_model(opt_val):
   model = keras.Sequential([
      layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
      ])

   optimizer = tf.keras.optimizers.Adam(opt_val)
   model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
   return model

model = build_model(OptVal) #Build model

EPOCHS = 100 #Number of epochs to train over
history = model.fit(
          normed_train_data, train_labels,
          epochs=EPOCHS, validation_split = 0.2, verbose=1,
          callbacks=[tfdocs.modeling.EpochDots()]) #Fit the model
hist = pd.DataFrame(history.history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
hist['epoch'] = history.epoch
if PLOT:
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    ax = plt.gca()
    line = ax.lines[0]
    data = line.get_xydata()
    np.save("Data/%s.npy"%Model_Name,data)
    plt.ylabel(r'MSE [AV$^2$]')
    plt.savefig("Data/%s.png"%Model_Name,dpi=600,bbox_inches='tight',transparent=True)
model.save('%s/my_model'%Model_Name) #Save model
keras.backend.clear_session()
