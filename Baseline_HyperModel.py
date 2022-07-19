from __future__ import print_function
import tensorflow as tf
import numpy as np
import Global_Config as gc
import csv

my_seed = 123
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf

tf.random.set_seed(my_seed)
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.utils import np_utils
from keras import callbacks
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
import keras.backend as K
import os
from keras import regularizers

SavedParameters = []


def NN(params):
    x_train = gc.train_X
    y_train = gc.train_Y
    print(params)

    input_shape = (x_train.shape[1],)
    input = Input(input_shape)
    l1 = Dense(params['neurons1'], activation='relu', kernel_initializer='glorot_uniform')(input)
    l1 = BatchNormalization()(l1)
    l1 = Dropout(params['dropout1'])(l1)
    l1 = Dense(params['neurons2'], activation='relu', kernel_initializer='glorot_uniform')(l1)

    softmax = Dense(gc.n_class, activation='softmax', kernel_initializer='glorot_uniform')(l1)

    adam = Adam(learning_rate=params['learning_rate'])
    model = Model(inputs=input, outputs=softmax)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)  # ,run_eagerly = True)

    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True), ]

    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                      test_size=0.2)  # before model building
    tic = time.time()

    YTraining = np_utils.to_categorical(YTraining, gc.n_class)
    YValidation = np_utils.to_categorical(YValidation, gc.n_class)

    h = model.fit(XTraining, YTraining, batch_size=params["batch"], epochs=150, verbose=2, callbacks=callbacks_list
                  , shuffle=True, validation_data=(XValidation, YValidation))

    toc = time.time()
    time_tot = toc - tic

    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)

    y_test = np.argmax(YValidation, axis=1)
    Y_predicted = model.predict(XValidation, verbose=0, use_multiprocessing=True, workers=12)

    Y_predicted = np.argmax(Y_predicted, axis=1)

    return model, h, {"val_loss": score,
                      "F1_MACRO": f1_score(y_test, Y_predicted, average='macro'),
                      "F1_MICRO": f1_score(y_test, Y_predicted, average='micro'),
                      "F1_WEIGHTED": f1_score(y_test, Y_predicted, average='weighted'),
                      "time": time_tot
                      }


def fit_and_score(params):
    model, h, val = NN(params)
    print("start predict")

    # print('y_test labels HYPER : ', gc.test_Y.value_counts())

    Y_predicted = model.predict(gc.test_X)
    Y_predicted = np.argmax(Y_predicted, axis=1)
    elapsed_time = val['time']

    K.clear_session()
    gc.SavedParameters.append(val)

    gc.SavedParameters[-1].update({"learning_rate": params["learning_rate"], "batch": params["batch"],
                                   "dropout1": params["dropout1"],
                                   "neurons_layer1": params["neurons1"],
                                   "neurons_layer2": params["neurons2"],

                                   "time": time.strftime("%H:%M:%S", time.gmtime(elapsed_time))})

    gc.SavedParameters[-1].update({
        "F1_MACRO_test": f1_score(gc.test_Y, Y_predicted, average='macro'),
        "F1_MICRO_test": f1_score(gc.test_Y, Y_predicted, average='micro'),
        "F1_WEIGHTED_test": f1_score(gc.test_Y, Y_predicted, average='weighted')})
    # Save model
    if gc.SavedParameters[-1]["val_loss"] < gc.best_score:
        print("new saved model:" + str(gc.SavedParameters[-1]))
        gc.best_model = model
        gc.best_score = gc.SavedParameters[-1]["val_loss"]
        print('validation best score : ', gc.best_score)
        gc.DNN_Model_Batch_Size = params['batch']

    SavedParameters = sorted(gc.SavedParameters, key=lambda i: i['val_loss'])

    try:
        with open(gc.test_path + 'Results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=SavedParameters[0].keys())
            writer.writeheader()
            writer.writerows(SavedParameters)
    except IOError:
        print("I/O error")
    return {'loss': val["val_loss"], 'status': STATUS_OK}  # cambia


def reset_global_variables(train_X, train_Y, test_X, test_Y):
    gc.train_X = train_X
    gc.train_Y = train_Y
    gc.test_X = test_X
    gc.test_Y = test_Y
    gc.DNN_Model_Batch_Size = 0
    gc.best_score = np.inf
    gc.best_scoreTest = 0
    gc.best_accuracy = 0
    gc.best_f1_macro = 0
    gc.best_f1_micro = 0
    gc.best_f1_weighted = 0
    gc.best_model = None
    gc.best_model_test = None
    gc.best_model_f1_micro = None
    gc.best_model_f1_macro = None
    gc.best_model_f1_weighted = None
    gc.best_time = 0
    gc.SavedParameters = []


def hypersearch(train_scaler, y_train, test_scaler, y_test, testPath, config_No):
    reset_global_variables(train_scaler, y_train, test_scaler, y_test)

    if gc.XAIFS == 'XAIFS':
        testPath = testPath + 'XAIFS_' + str(config_No)
    else:
        testPath = testPath + 'ConfigNo' + str(config_No)

    try:
        os.remove(testPath + 'Results.csv')
    except IOError:
        print('The file has been deleted')

    gc.test_path = testPath
    print('test path : ', gc.test_path)

    space = {"batch": hp.choice("batch", [32, 64, 100, 128, 256, 512, 1024]),
             'dropout1': hp.uniform("dropout1", 0, 1),

             "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),

             "neurons1": hp.choice("neurons1", [32, 64, 128, 256, 512, 1024]),
             "neurons2": hp.choice("neurons2", [32, 64, 128, 256, 512, 1024]),

             }

    trials = Trials()

    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=1, trials=trials,
                rstate=np.random.RandomState(my_seed))

    return gc.best_model, gc.best_time, gc.best_score

