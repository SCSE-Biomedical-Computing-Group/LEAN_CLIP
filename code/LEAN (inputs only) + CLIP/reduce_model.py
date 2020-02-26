"""
This file contains most of the code involved in computing the important features

main.py calls the compute_deeplift_scores function, which in turn calls the other functions in this file.
"""
import os
import re

import numpy as np
from scipy import stats

import powerlaw as pl

import keras
from keras.layers import (Activation, Dense, Dropout)

import deeplift
from deeplift.conversion import kerasapi_conversion as kc

from utils import corr_mx_flatten_single, mkdir, plot_hist


def create_matrix(flat_list):
    """
    Given a flat list, converts it into a matrix (i.e. unflattens it)

    Inputs:
    - flat_list: 1-d Numpy array 
    
    Returns:
    - matrix: 2-d Numpy array of flat_list
    """

    matrix = np.zeros((264, 264))
    matrix[np.triu_indices(264, 1)] = flat_list
    matrix_T = matrix.T
    matrix = matrix + matrix_T - np.diag(np.diag(matrix_T))

    return matrix

def get_masked_data(data, mask): 
    """
    Applies the current mask to the data to get a reduced set of features
    Needed when repeatedly reducing the model. In this case, mask is an array of 1s so it does noto make a difference

    Inputs:
    - data: list of Numpy arrays containing the data matrices # data.shape is (num samples, num features)
    - mask: Numpy array of 1s and 0s, with 1 representing a selected feature

    Returns:
    - masked_data_array: Numpy array of the data after the mask is applied
    - mapping: a dictionary keeping a map from the original indez to the new index in the masked data
    """

    masked_data = []
    mapping = dict() # key is the original index, value is the new index
    new_index = 0

    for orig_index, value in enumerate(mask):
        if value == 1:
            mapping[orig_index] = new_index
            new_index += 1
        else:
            mapping[orig_index] = -1
    
    for sample in data:
        masked_sample = np.zeros(int(np.sum(mask)))
        for orig_index in mapping.keys():
            new_index = mapping[orig_index]
            if new_index != -1:
                masked_sample[new_index] = sample[orig_index]

        masked_data.append(masked_sample)

    masked_data_array = np.array(masked_data)

    return masked_data_array, mapping

def get_padded_data(data, mapping):
    """
    Gets a reduced dataset back to the original dimensions
    Used when repeatedly removing neurons

    Inputs: 
    - data: list of Numpy arrays containing the data matrices # data.shape is (num samples, num features)
    - mapping: a dictionary keeping a map from the original indez to the new index in the masked data

    Returns:
    - padded_data: Numpy array of data reshaped to original data dimension (before mask)
    """

    padded_data = np.zeros(len(mapping))
    for orig_index in mapping.keys():
        new_index = mapping[orig_index]
        if new_index != -1:
            padded_data[orig_index] = data[new_index]

    return padded_data

def get_reference(mode, reference_label, data, data_labels, each_sample_index = 0):
    """
    Computes the reference to be used by DeepLIFT later

    Inputs:
    - mode: Type of reference computation to perform, one of 'avearge' 'each' 'representative' 'best' (string)
    - reference_label: label ID that is used as the reference class in data_labels (i.e. Y)
    - data: Numpy array of the data matrices
    - data_labels: Numpy array of data labels (i.e. Y)
    - each_sample_index = 0

    Returns:
    - rep_reference: Numpy array of reference, modified from data, to be used by DeepLIFT
    """
    reference = np.zeros(data[0].shape)
    num_reference_samples = np.sum(data_labels == reference_label) 

    average_reference, norm_reference, modal_reference = np.zeros(data[0].shape), np.zeros(data[0].shape), np.zeros(data[0].shape)

    print('mode is', mode)
    
    if mode == 'average' or mode == 'best':
        for index, x in enumerate(data_labels):
            if x == reference_label:
                reference += data[index]
        reference = reference/num_reference_samples
        
        average_reference = np.copy(reference)

    if mode == 'each':
        for index, sample in enumerate(data[data_labels == reference_label]):
            if index == each_sample_index:
                reference = sample
    
    if mode == 'representative' or mode == 'best':
        similarity_scores = dict()
        for index_1, sample_1 in enumerate(data[data_labels == reference_label]):
            similarity_scores[index_1] = 0
            for index_2, sample_2 in enumerate(data[data_labels == reference_label]):
                 similarity_scores[index_1] += np.linalg.norm(sample_1 - sample_2)

        reference_index = min(similarity_scores, key=similarity_scores.get)
        reference = data[data_labels == reference_label][reference_index]

        norm_reference = np.copy(reference)

    if mode == 'modal' or mode == 'best':
        reference_data = data[data_labels == reference_label]

        for i in range(reference_data.shape[1]):
            vals= np.around(reference_data[:, i], decimals=2)
            reference[i], _  = stats.mode(vals, nan_policy = 'omit')

        modal_reference = np.copy(reference)

    if mode == 'best':
        similarity_scores = dict()
        references_dict = {'average': average_reference, 'norm': norm_reference, 'modal': modal_reference}
        
        for ref in references_dict.keys():
            similarity_scores[ref] = 0
            for index, sample in enumerate(data[data_labels == reference_label]):
                 similarity_scores[ref] += np.linalg.norm(references_dict[ref] - sample)
                 print(ref, index, similarity_scores[ref], np.linalg.norm(references_dict[ref] - sample))

        best_reference = min(similarity_scores, key=similarity_scores.get)
        reference = references_dict[best_reference]

        print(mode, similarity_scores, best_reference)
    
    num_copies = data.shape[0] - np.sum(data_labels == reference_label) 
    
    rep_reference = []
    for copy in range(num_copies):
        rep_reference.append(reference)

    return rep_reference

def one_shot_removal(feature_score, alpha):
    """
    Fits the distribution of saliency score to various distributions, find the best fitting one and keep alpha % of the features
    Performed for a single layer; this function is called by called by compute_new_reduced_model 
    Inputs:
    - feature_score: Numpy array containing the saliency score for each feature
    - alpha: 1 - alpha represents the fraction of (the most important) features to keep (float)

    Returns: 
    - selected_features: Numpy array containing 1s and 0s, 1 represents a selected feature 
    """

    selected_features = np.zeros(np.shape(feature_score))

    LAYER_SIZE_THRESHOLD = 2 

    if np.shape(feature_score)[0] > LAYER_SIZE_THRESHOLD:
        feature_score[feature_score == 0] = 1e-10
        x_min = np.min(feature_score) 
        x_max = np.max(feature_score) 
        params_power_law, loglikelihood_power_law = pl.distribution_fit(np.asarray(feature_score), distribution='power_law', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)
        params_lognormal, loglikelihood_lognormal = pl.distribution_fit(np.asarray(feature_score), distribution='lognormal', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)
        params_expo, loglikelihood_expo = pl.distribution_fit(np.asarray(feature_score), distribution='exponential', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)
        params_stretched, loglikelihood_stretched = pl.distribution_fit(np.asarray(feature_score), distribution='stretched_exponential', xmin=x_min, xmax=x_max, discrete=False, comparison_alpha=False, search_method='Likelihood', estimate_discrete=False)

        print('Shape of layer', np.shape(feature_score))
        print('loglikelihood_power_law', loglikelihood_power_law, 'loglikelihood_lognormal', loglikelihood_lognormal, 'loglikelihood_expo', loglikelihood_expo, 'loglikelihood_stretched', loglikelihood_stretched) 

        if loglikelihood_power_law > max(loglikelihood_lognormal, loglikelihood_expo, loglikelihood_stretched):  
            theoretical_distribution = pl.Power_Law(xmin=x_min, parameters=params_power_law, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Power_Law'
            best_param = params_power_law

        elif loglikelihood_lognormal > max(loglikelihood_power_law, loglikelihood_expo, loglikelihood_stretched):
            theoretical_distribution = pl.Lognormal(xmin=x_min, parameters=params_lognormal, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Lognormal'
            best_param = params_lognormal

        elif loglikelihood_expo > max(loglikelihood_power_law, loglikelihood_lognormal, loglikelihood_stretched):
            theoretical_distribution = pl.Exponential(xmin=x_min, parameters=params_expo, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Exponential'
            best_param = params_expo

        elif loglikelihood_stretched > max(loglikelihood_power_law, loglikelihood_lognormal, loglikelihood_expo):
            theoretical_distribution = pl.Stretched_Exponential(xmin=x_min, parameters=params_stretched, xmax=x_max, discrete=False)
            prob_dist = theoretical_distribution.cdf(feature_score)
            best_fit_dist = 'Stretched_Exponential'
            best_param = params_stretched

        print('values', feature_score)
        print('PDF: ', prob_dist, prob_dist.shape, 'best fit distribution', best_fit_dist, 'best params ', best_param)
        selected_features = prob_dist > (1 - alpha)
        
        print('Number of DeepLIFT selected features: ', np.sum(selected_features))

    if np.shape(feature_score)[0] < LAYER_SIZE_THRESHOLD or np.sum(selected_features) == 0:
        selected_features = np.ones(np.shape(feature_score))

    return selected_features
 
def compute_new_reduced_model(model, dropout, layer_scores, number_output_neurons, alpha, mapping, cluster_mask, flags):
    """
    Generates 

    Inputs: 
    - model: Keras model of the original best model
    - dropout: fraction of nodes to turn off (float)
    - layer_scores: list containing Numpy arrays of saliency scores for each layer
    - number_output_neurons: number of classes in dataset (int)
    - alpha: 1 - percentage of features to keep (float)
    - mapping: dictionary mapping indices from original matrix to reduce matrix
    - cluster_mask: Numpy array containing mask obtained from CLIP
    - flags: contains model settings (dict)

    Returns:
    - new_model: Keras model obtained after model pruning
    - matrix_mask: Numpy array containing feature mask obtained after LEAN/CLIP
    """

    ## Input layer / features

    input_layer_scores = layer_scores[0]
    flat_mask_reduced = one_shot_removal(input_layer_scores, alpha)
    merged_masks = np.logical_or(flat_mask_reduced, cluster_mask).astype(int) # assuming for both, 1 means keep 
    merged_masks_intersection = np.logical_and(flat_mask_reduced, cluster_mask).astype(int)
    print("Number of overlapping features between DeepLIFT and Correlation: " + str(np.sum(merged_masks_intersection)))

    if flags['USE_DEEPLIFT_FEATURES'] and flags['USE_CORRELATION_FEATURES']:
        flat_mask = get_padded_data(merged_masks, mapping)
        importance_score_mask_before = merged_masks
    elif not flags['USE_DEEPLIFT_FEATURES'] and flags['USE_CORRELATION_FEATURES']:
        print("Not using DeepLIFT")
        flat_mask = get_padded_data(cluster_mask, mapping)
        importance_score_mask_before = cluster_mask
    elif not flags['USE_CORRELATION_FEATURES'] and flags['USE_DEEPLIFT_FEATURES']:
        print("Not using Correlation features")
        flat_mask = get_padded_data(flat_mask_reduced, mapping)
        importance_score_mask_before = flat_mask_reduced
    else:
        print("Not filtering...")
        flat_mask = get_padded_data(merged_masks, mapping) 
        importance_score_mask_before = merged_masks 
    
    matrix_mask = create_matrix(flat_mask)

    ## Hidden layers

    new_model = keras.models.Sequential()

    idx_new_model = 0 # captures the model layer ID in the new model

    for idx, layer in enumerate(model.layers[0:-2]): # first hidden layer to before the output layer (dense + activation)
        if type(layer) == Dense:
            print('***************** Defining new model Layer *************', idx_new_model)
            
            print(type(model.layers[idx + 1]), type(model.layers[idx + 2])) 
            if type(model.layers[idx + 1]) == Dense and len(layer_scores[idx + 1]) > number_output_neurons:
                if flags['PERFORM_PRUNING']:
                    importance_score_mask_present = one_shot_removal(layer_scores[idx + 1], alpha)
                else:
                    importance_score_mask_present = np.ones(len(layer_scores[idx + 1]))
                state = 1
                print('1', idx + 1, layer_scores[idx + 1])

            elif type(model.layers[idx + 2]) == Dense and len(layer_scores[idx + 2]) > number_output_neurons: 
                if flags['PERFORM_PRUNING']:
                    importance_score_mask_present = one_shot_removal(layer_scores[idx + 2], alpha)
                else:
                    importance_score_mask_present = np.ones(len(layer_scores[idx + 2]))
                state = 2
                print('2', idx + 2, layer_scores[idx + 2])

            elif type(model.layers[idx + 1]) == Dense and len(layer_scores[idx + 1]) <= number_output_neurons: 
                importance_score_mask_present = np.ones(len(layer_scores[idx + 1]))
                state = 1
                print('3', idx + 1, layer_scores[idx + 1]) # intermediate layers shouldn't have lesser neurons than output layer
            
            elif type(model.layers[idx + 2]) == Dense and len(layer_scores[idx + 2]) <= number_output_neurons: 
                importance_score_mask_present = np.ones(len(layer_scores[idx + 2]))
                state = 2
                print('4', idx + 2, layer_scores[idx + 2])

            layer_weights = model.get_layer(model.layers[idx].name).get_weights()[0] # 0 is for weights 1 for bias
            new_weights = layer_weights[np.ix_(np.where(importance_score_mask_before > 0)[0], np.where(importance_score_mask_present > 0)[0])]
            
            print('original number of features of preceding layer:', importance_score_mask_before.shape)
            print('new number of features of preceding layer: ', np.sum(importance_score_mask_before)) 
            print('original number of features of this layer:', importance_score_mask_present.shape)
            print('new number of features of this layer:',  np.sum(importance_score_mask_present))
            print('Weight dimensions: old', layer_weights.shape, 'new ', new_weights.shape)
            
            layer_bias = model.get_layer(model.layers[idx].name).get_weights()[1]
            new_bias = layer_bias[np.where(importance_score_mask_present > 0)[0]]
            
            print('Bias dimensions: old', layer_bias.shape, 'new ', new_bias.shape)

            new_number_of_nodes = int(np.sum(importance_score_mask_present))

            if flags['TRAIN_FROM_SCRATCH']:

                if idx_new_model == 0: # input layer to first hidden layer
                    new_dense_layer = Dense(new_number_of_nodes, activation = 'relu', input_dim= int(np.sum(importance_score_mask_before))) # use default weights and bias
                else: # 2nd hidden layer onwards
                    new_dense_layer = Dense(new_number_of_nodes, activation = 'relu')

            elif not flags['TRAIN_FROM_SCRATCH']:

                if idx_new_model == 0: # input layer to first hidden layer
                    new_dense_layer = Dense(new_number_of_nodes, activation = 'relu', input_dim= int(np.sum(importance_score_mask_before)), weights = [new_weights, new_bias])
                else: # 2nd hidden layer onwards
                    new_dense_layer = Dense(new_number_of_nodes, activation = 'relu', weights = [new_weights, new_bias])

            new_model.add(new_dense_layer)
            new_model.add(Dropout(rate=dropout))
            importance_score_mask_before = importance_score_mask_present
            idx_new_model +=1

    # Output layer should have same number of output neurons, although the previous layer's neurons may have changed. Therefore the weight matrix is updated, but bias is same. 
    output_layer_weights = model.get_layer(model.layers[-2].name).get_weights()[0] # 0 is for weights 1 for bias
    new_output_layer_weights = output_layer_weights[np.where(importance_score_mask_before > 0)[0]]
    
    output_layer_bias = model.get_layer(model.layers[-2].name).get_weights()[1]

    if flags['TRAIN_FROM_SCRATCH']:
        new_output_layer = Dense(number_output_neurons) # use default weights and bias
    elif not flags['TRAIN_FROM_SCRATCH']:
        new_output_layer = Dense(number_output_neurons, weights = [new_output_layer_weights, output_layer_bias])
        
    new_model.add(new_output_layer)
    new_model.add(Activation('softmax'))
    print(new_model.summary())

    return new_model, matrix_mask

def compute_deeplift_scores(
    TARGET_DIRECTORY, dataset, X, Y, 
    keras_model_file, reference_label, non_reference_label, base_neuron_label, 
    mask, gpu_id, dropout, threshold, percentage_cutoff, cluster_mask,
    flags):
    """
    Wrapper function for model reduction, called by main.py
    Uses DeepLIFT to compute saliency scores for feature selection, with the average data used as reference
    See https://github.com/kundajelab/deeplift for DeepLIFT implementation

    Inputs:
    - TARGET_DIRECTORY: general directory path to write files to (str)
    - dataset: choice of dataset, along with seed and fold number (str) 
    - X: Numpy array containing data matrices
    - Y: Numpy array containing data labels
    - keras_model_file: name of existing model file (str)
    - reference_label: how the reference class is represented in Y, usually 0 (int)
    - non_reference_label: how the other class(es) is (are) represented in Y, usually 1 (int)
    - base_neuron_label: label to be used as the base, usually 0 (int)
    - mask: Numpy array, usually initialised as all 1s unless neurons are repeatedly removed
    - gpu_id: ID of GPU to use (int)
    - dropout: fraction of neurons to turn off (float)
    - threshold: usually set as 1.0, represents the previous percentage_cutoff when repeatedly removing neurons (float)
    - percentage_cutoff: usually set as 0.95 to keep 5% of the most significant features (float)
    - cluster_mask: Numpy array containing mask obtained from CLIP
    - flags: used to vary model settings, see main.py (dict)

    Returns:
    - new_model_file: directory path to the new model (str)
    - mask_2D_flattened: Numpy array of 1s and 0s, with 1 representing a selected feature
    """

    keras_model = keras.models.load_model(keras_model_file)
    print(keras_model.summary()) # original model

    deeplift_model = kc.convert_model_from_saved_files(
        keras_model_file, 
        nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)
    print(deeplift_model.get_layers())

    mode = 'average'
    X_masked, mapping = get_masked_data(X, mask)
    Y = np.argmax(Y, axis=1)
    reference = get_reference(mode, reference_label, X_masked, Y)
    
    print('+++++++++++++ Computing DeepLIFT scores ++++++++++++++')
    print('previous threshold', threshold, 'new threshold', percentage_cutoff)

    find_scores_layer_idx = 0
    input_scores = np.zeros(X_masked[Y == non_reference_label].shape)
    layer_scores = []
    task_id = base_neuron_label

    for layer_idx, layer in enumerate(deeplift_model.get_layers()):
        if type(layer).__name__ == 'Dense' or type(layer).__name__ == 'Input':
            deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=layer_idx, target_layer_idx=-2)
            scores = np.array(deeplift_contribs_func(task_idx=task_id,
                                                     input_references_list=reference,
                                                     input_data_list=[X_masked[Y == non_reference_label]],
                                                     batch_size=10,
                                                     progress_update=50))
                                                     
            sum_scores = np.zeros(scores.shape[1])
            
            for score in scores:
                sum_scores += score
            
            sum_scores = np.absolute(sum_scores)

            if sum_scores.shape[0] > 2:
                plot_hist(TARGET_DIRECTORY, sum_scores, dataset + '_t_' + str(threshold) + '_layer_' + str(layer_idx)) 
            
            print('layer', layer_idx, 'type is: ', type(layer), 'scores dimensions are: ', scores.shape, 'sum_scores', sum_scores.shape)
            layer_scores.append(sum_scores)
            
            if layer_idx == 0:
                input_scores = np.square(scores)
                layer_scores.append([])

        elif type(layer).__name__ == 'NoOp' or type(layer).__name__ == 'Softmax':
            layer_scores.append([])
            print('layer', layer_idx, 'type is: ', type(layer).__name__)

    alpha = (1 - (percentage_cutoff/threshold))
    new_model, mask_2D = compute_new_reduced_model(keras_model, dropout, layer_scores, 2, alpha, mapping, cluster_mask, flags)

    input_sum_scores = np.zeros(X_masked.shape[1])

    for input_score in input_scores:
        input_sum_scores += input_score

    padded_sum_scores = get_padded_data(input_sum_scores, mapping)
    full_matrix = create_matrix(padded_sum_scores)

    mkdir(TARGET_DIRECTORY + './important_features/')
    np.savetxt(
        TARGET_DIRECTORY + './important_features/' + dataset + '_scores_deeplift_reduced_r_' + str(threshold) + '_t_' + str(percentage_cutoff) + '.csv', 
        np.transpose(np.array(input_scores)), delimiter= ',')
    np.savetxt(
        TARGET_DIRECTORY + './important_features/' + dataset + '_scores_reshaped_reduced_r_' + str(threshold) + '_t_' + str(percentage_cutoff) + '.csv', 
        full_matrix, delimiter=",")
    np.savetxt(
        TARGET_DIRECTORY + './important_features/' + dataset + '_deeplift_features_nodes_r_' + str(threshold) + '_t_' + str(percentage_cutoff) + '.csv', 
        mask_2D)
    
    mkdir(TARGET_DIRECTORY + './reduced_models/')
    new_model_file = TARGET_DIRECTORY + './reduced_models/' + dataset + '_from_' + str(threshold) + '_to_' + str(percentage_cutoff) + '.h5'
    new_model.save(new_model_file)

    os.remove(keras_model_file)

    mask_2D_flattened = corr_mx_flatten_single(mask_2D)
    
    return new_model_file, mask_2D_flattened
