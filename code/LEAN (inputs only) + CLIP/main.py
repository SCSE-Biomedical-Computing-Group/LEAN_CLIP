"""
Main script for LEAN (inputs only) + CLIP
"""

import os
import sys

import h5py

import numpy as np
import scipy
import tensorflow as tf

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten, LocallyConnected2D, MaxPooling2D)

from sklearn import metrics

from reduce_model import compute_deeplift_scores
from utils import corr_mx_flatten, mkdir, prepare_dataset, split_kfoldcv_sbj


## Functions

def funcNetFFN_2L(input_shape, num_neurons, dropout, batch_size, file):
    """
    Model used for ABIDE

    Inputs:
    - input_shape: tuple of the input dimensions, required for creating a Dense layer in Keras
    - num_neurons: number of neurons in the first layer (int)
    - dropout: fraction of the nodes to drop (float)
    - batch_size (int)
    - file: filename for the model, for saving (str)

    Returns
    - model: A Keras model
    """

    model = keras.models.Sequential()

    model.add(Dense(num_neurons, activation='relu', input_dim=input_shape))
    model.add(Dropout(rate=dropout))

    model.add(Dense(10, activation='relu'))
    model.add(Dropout(rate=dropout))
    
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.save(file)
    
    return model

def funcNetFFN_3L(input_shape, num_neurons, dropout, batch_size, file):
    """
    Model used for all disorders except ABIDE (see funcNetFFN_2L for ABIDE)

    Inputs:
    - input_shape: tuple of the input dimensions, required for creating a Dense layer in Keras
    - num_neurons: number of neurons in the first layer (int)
    - dropout: fraction of the nodes to drop (float)
    - batch_size (int)
    - file: filename for the model, for saving (str)

    Returns
    - model: A Keras model
    """

    model = keras.models.Sequential()

    model.add(Dense(num_neurons, activation='relu', input_dim=input_shape))
    model.add(Dropout(rate=dropout))
    
    if num_neurons <= 50:
        model.add(Dense(32, activation='relu'))
    elif num_neurons > 50:
        model.add(Dense(64, activation='relu'))    
    model.add(Dropout(rate=dropout))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=dropout))
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.save(file)
    
    return model


## Configuration to run (dataset)

class_subset = str(sys.argv[1]) # 'CN-AD' 'CN-MCI' 'MDD' 'ADHD' 'ABIDE' 

mask_initial = np.ones((264, 264))
(subject_groups_, X__, Y__, hidden_layer_size, cnn_model) = prepare_dataset(class_subset, mask_initial)

flags = {
    
    'USE_CORRELATION_FEATURES' : True,
    'USE_DEEPLIFT_FEATURES' : True,
    'PERFORM_PRUNING' : False,
    
    'EARLY_STOPPING' : True,
    'TRAIN_FROM_SCRATCH' : True,
}
print(flags)

## Directory Setup 

SOURCE_DIRECTORY = '../../data/cluster_mask/' + class_subset + '/' 

TARGET_DIRECTORY = './results/model_results_' 
if flags['TRAIN_FROM_SCRATCH']:
    TARGET_DIRECTORY += 'trainfromscratch_lean_'
if flags['USE_CORRELATION_FEATURES']:
    if flags['PERFORM_PRUNING']:
        TARGET_DIRECTORY += 'pruning_'
    else:
        TARGET_DIRECTORY += 'nopruning_'
if flags['EARLY_STOPPING']:
    TARGET_DIRECTORY += 'es'
TARGET_DIRECTORY += '/' + class_subset + '/' + cnn_model + '/'

mkdir(TARGET_DIRECTORY)
mkdir(TARGET_DIRECTORY + "best_model/")
mkdir(TARGET_DIRECTORY + "importance_scores/")
mkdir(TARGET_DIRECTORY + "initial_model/")

## Model Parameters

epochs = 200 
batch_size = 8
learning_rate = 0.0001
decay = 0.001
dropout = 0.1

## Misc settings

seeds = range(10,20)
folds = 5 
mode = 'modular'
threshold = 0.95

gpu_id = str(sys.argv[2]) # 0, 1, 2, 3
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

## Main Code

with open(TARGET_DIRECTORY + cnn_model + '_' + mode + '_kfold_training_logs_' + class_subset + '.csv', 'a') as out_stream:
    out_stream.write('Seed,Threshold,Fold,Best Epoch,Training Accuracy,Test Accuracy,Training Accuracy - Test Accuacy,Train Loss,Test Loss,Training Loss - Test Loss,MAE,AUC,Trainable Parameters\n')

for SEED in seeds:
    
    np.random.seed(SEED)
    idx = np.arange(len(X__))
    np.random.shuffle(idx) # randomize index
    X, Y, subject_groups = X__[idx], Y__[idx], np.array(subject_groups_)[idx] 
    subject_groups = subject_groups.tolist()
    
    folds_indices = split_kfoldcv_sbj(Y.argmax(1), subject_groups, folds, SEED)

    fold_count = 0

    for train_index, val_index in folds_indices: # for each fold

        if 'best_model_seed_' + str(SEED) + '_' + str(class_subset) + '_' + cnn_model + '_' + mode + '_' + str(threshold) + '_fold_' + str(fold_count) + '.h5' in os.listdir(TARGET_DIRECTORY + 'best_model/' ):
            print('SEED_' + str(SEED) + '_fold_' + str(fold_count) + ' done, skipping it....')
            fold_count += 1
            continue

        print('************** Start of Fold:', fold_count, '**********************')
        thresholds = np.array([1.0, float(threshold)]) 
        mask = np.ones(34716)

        for ix, i in enumerate(thresholds): 

            if ix == 0:
                model_initial_file = TARGET_DIRECTORY + 'initial_model/' + 'initial_' + cnn_model + '_' + str(SEED) + '_' + str(class_subset) + '_' + str(i) + '.h5'
                X_ = corr_mx_flatten(X)
                if cnn_model == 'funcNetFFN_2L':
                    _ = funcNetFFN_2L(X_.shape[1], hidden_layer_size, dropout, batch_size, model_initial_file)
                elif cnn_model == 'funcNetFFN_3L':
                    _ = funcNetFFN_3L(X_.shape[1], hidden_layer_size, dropout, batch_size, model_initial_file)

                print ('Shape of input data', X_.shape)

            else:
                threshold_1 = thresholds[ix-1]
                threshold_2 = thresholds[ix]
                prev_threshold = thresholds[ix-1]
                prev_best_model = TARGET_DIRECTORY + 'best_model/' + 'best_model_seed_' + str(SEED) + '_' + str(class_subset) + '_' + cnn_model + '_' + mode + '_' + str(prev_threshold) + '_fold_' + str(fold_count) +  '.h5'
                X_flat = corr_mx_flatten(X)
                
                model_initial_file, mask = compute_deeplift_scores(
                    TARGET_DIRECTORY, class_subset + '_SEED_' + str(SEED) + '_fold_' + str(fold_count), X_flat[train_index], Y[train_index], 
                    prev_best_model, 0, 1, 0, mask, gpu_id, dropout, threshold_1, threshold_2, cluster_mask, flags)
                X_ = []
                for matrix in X_flat:
                    masked_matrix = np.multiply(matrix, mask)
                    X_.append(masked_matrix[mask == 1])
                X_ = np.array(X_)
                print('Shape of input data', X_.shape)

            X_train, Y_train = X_[train_index], Y[train_index]
            X_val, Y_val = X_[val_index], Y[val_index]

            if ix == 0: # done in the first iteration

                cluster_mask = np.loadtxt(SOURCE_DIRECTORY + class_subset + '_cluster_mask_seed' + str(SEED) + '_fold_' + str(fold_count) + '.csv').astype(int)
        
            print('************** Fold:', fold_count, '**********************')
            
            subject_groups_temp = np.array(subject_groups)
            
            Adam = keras.optimizers.Adam(lr=learning_rate) 
            model = keras.models.load_model(model_initial_file)
            model.compile(loss= "categorical_crossentropy", optimizer=Adam, metrics=["accuracy"])

            model_filename = TARGET_DIRECTORY + 'best_model/' + 'best_model_seed_' + str(SEED) + '_' +  str(class_subset) + '_' + cnn_model + '_' + mode + '_' + str(i) + '_fold_' + str(fold_count) + '.h5'
            
            if flags['EARLY_STOPPING']:
                callbacks = [ModelCheckpoint(filepath=model_filename, monitor='val_acc', save_best_only=True), EarlyStopping(monitor='val_loss', patience=10)] 
            else: 
                callbacks = [ModelCheckpoint(filepath=model_filename, monitor='val_acc', save_best_only=True)]

            # Train model
            history = model.fit(X_train, Y_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs, validation_data=(X_val, Y_val), verbose=0)
            score = model.evaluate(X_val, Y_val, batch_size=batch_size)

            # Retrieve results 
            loss_values = history.history['loss']
            acc_values = history.history['acc']
            valloss_values = history.history['val_loss']
            valacc_values = history.history['val_acc']
                        
            best_val_acc = max(history.history['val_acc'])
            best_epoch = history.history['val_acc'].index(max(history.history['val_acc']))
            
            acc_at_best_val_acc_epoch = history.history['acc'][best_epoch]            
            overfitting_metric = acc_at_best_val_acc_epoch - best_val_acc

            loss_at_best_val_acc_epoch = history.history['loss'][best_epoch]
            val_loss_at_best_val_acc_epoch = history.history['val_loss'][best_epoch]
            overfitting_metric_2 = val_loss_at_best_val_acc_epoch - loss_at_best_val_acc_epoch
            model.load_weights(model_filename)

            best_prediction = model.predict(X_val, batch_size=batch_size, verbose=0)
            MAE = metrics.mean_absolute_error(Y_val, best_prediction, sample_weight=None, multioutput='uniform_average')
            AUC = metrics.roc_auc_score(Y_val, best_prediction, average='macro', sample_weight=None, max_fpr=None)

            print(SEED, 'fold', fold_count, 'accuracy ', best_val_acc, 'loaded: ', model.evaluate(X_val, Y_val, batch_size=batch_size), 'at epoch ', best_epoch, 'MAE ', MAE, 'AUC ', AUC)

            all_trainable_count = int(np.sum([keras.backend.count_params(p) for p in set(model.trainable_weights)]))

            layers_num_neurons = []
            for idx, layer in enumerate(model.layers[0:-2]):
                layers_num_neurons.append(layer.input_shape[1])

            with open(TARGET_DIRECTORY + cnn_model + '_' + mode + '_kfold_training_logs_' + class_subset + '.csv', 'a') as out_stream:
                out_stream.write(
                    str(SEED) + ', ' + str(i) + ', ' + str(fold_count) + ', ' + str(best_epoch) + \
                    ', ' + str(acc_at_best_val_acc_epoch) + ', ' + str(best_val_acc) + ', ' + \
                    str(overfitting_metric) + ', ' + str(loss_at_best_val_acc_epoch) + ', ' + \
                    str(val_loss_at_best_val_acc_epoch) + ', ' + str(overfitting_metric_2) + ', ' + \
                    str(MAE) + ', ' + str(AUC) + ', ' + str(all_trainable_count) + ', ' +  \
                    str(layers_num_neurons) + '\n')

            keras.backend.clear_session()

        fold_count += 1
 