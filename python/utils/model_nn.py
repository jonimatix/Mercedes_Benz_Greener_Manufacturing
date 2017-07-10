import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error

# Keras is a deep learning library that wraps the efficient numerical libraries Theano and TensorFlow.
# It provides a clean and simple API that allows you to define and evaluate deep learning models in just a few lines of code.from keras.models import Sequential, load_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
# define custom R2 metrics for Keras backend
from keras import backend as K
# to tune the NN
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
import h5py

# debug
import pdb

# define path to save model
import os

# r_2 for nn
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# keras
def model_nn(input_dims = 678, dropout_level = 0.25, activation = 'tanh'):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dims, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(1024, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(1024, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(1024, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(1024, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(768, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(768, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(768, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(768, kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))

    model.add(Dense(768, kernel_initializer="he_normal", kernel_regularizer = l2(1.e-5)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_level))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss = "mean_squared_error", # one may use 'mean_absolute_error' as alternative
                  optimizer = "adam",
                  metrics = [r2_keras, "accuracy"] # you can add several if needed
                 )
    
    # Visualize NN architecture
#    print(model.summary())
    return model

def model_nn_estimator(X_train, y_train, X_valid, y_valid, X_test):
       
    input_dims = X_train.shape[1]
#    X_train = X_train.as_matrix()
#    X_valid = X_valid.as_matrix()
#    X_test = X_test.as_matrix()
    
    model_path = "/media/noahhhhhh/dataScience/proj/competition/data/Mercedes_Benz_Greener_Manufacturing/model/model_nn.h5"
    
    estimator = KerasRegressor(
        build_fn = model_nn, 
        input_dims = input_dims,
        nb_epoch = 300, 
        batch_size = 35,
        verbose = 0
    )
    
    # prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor = 'val_loss', 
            patience = 20,
            verbose = 1),
        ModelCheckpoint(
            model_path, 
            monitor = 'val_loss', 
            save_best_only = True, 
            verbose = 0)
    ]
    
    # fit estimator
    history = estimator.fit(
        X_train, 
        y_train, 
        batch_size = 64,
        epochs = 500,
        validation_data = (X_valid, y_valid),
        verbose = 2,
        callbacks = callbacks,
        shuffle = True
    )
    
    if os.path.isfile(model_path):
        history.model.load_weights(model_path)
    
    pred_valid = history.model.predict(X_valid)
    pred_test = history.model.predict(X_test)
    return ([i for [i] in pred_valid], [i for [i] in pred_test])