"""
Utility functions
"""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter("ignore", UserWarning)

import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt
from pylab import savefig

## Correlation Matrix Manipulation

def corr_mx_flatten(X):
    """
    Takes in a list of matrices, flattens them and returns a Numpy array of flattened matrices

    Inputs:
    - X: List or Numpy array of matrices

    Return:
    - X_flattened: Numpy array of flattened matrices
    """

    num_features = int((X.shape[1]) * (X.shape[1] - 1) * 0.5)
    X_flattened = np.empty((X.shape[0], num_features))

    for i, matrix in enumerate(X):
        matrix_upper_triangular = matrix[np.triu_indices(np.shape(matrix)[0],1)]
        X_flattened[i] = np.ravel(matrix_upper_triangular, order="C")

    return X_flattened

def corr_mx_flatten_single(matrix):
    """
    Takes in a single matrix and returns a flattened version of it

    Inputs:
    - matrix: Numpy array containing a single matrix 

    Returns:
    - matrix_flat: Numpy array of the flattened matrix
    """

    matrix_upper_triangular = matrix[np.triu_indices(np.shape(matrix)[0], 1)]
    matrix_flat = np.ravel(matrix_upper_triangular, order="C")

    return matrix_flat

## Mapping

def get_keep_indices_from_mapping(mapping, indices_to_keep): 
    """
    Used for CLIP, uses mapping to get the indices of the original corr_matrix given the indices from the smaller matrix (due to feature reduction)
    
    Inputs
    - mapping: A dictionary containing the mapping from the original corr_matrix to the smaller one 
    - indices_to_keep: Numpy array of the new indices which need to be kept

    Returns
    - Numpy array of the old indices 
    """

    old_indices = [] # from the 34716 vector

    for new_index in indices_to_keep:
        old_indices.append(mapping[new_index])

    old_indices_array = np.array(old_indices)

    return old_indices_array

def map_main_to_sub(orig_indices):
    """
    Used for CLIP, maps the indices from the original corr_matrix to the smaller one (due to the feature reduction)
    
    In mapping, 
    - Keys = new indices (in sub_array)
    - Values = original indices (from 34716)
    
    Inputs:
    - orig_indices: Numpy array of indices, stores True if matches cluster_id (see generate_cluster_masks in generate_CLIP_features.py)

    Returns
    - mapping: A dictionary containing the mapping from the original corr_matrix to the smaller one
    """

    mapping = dict()

    for key in range(len(orig_indices)):
        mapping[key] = orig_indices[key]
    
    return mapping

## Directory Setup

def mkdir(dir):
    """
    Given a directory, creates it if it doesn't exist (can be nested)

    Input:
    - dir: Path for directory to create (string)
    """

    if not os.path.isdir(str(dir)):
        print("Folder does not exist yet. Creating the folder " + str(dir))
        os.makedirs(str(dir))

## Plots

def plot_hist(TARGET_DIRECTORY, data_flat, suffix):
    """
    Creates a histogram of salience scores

    Inputs:
    - TARGET_DIRECTORY: path to write the historgram to (str)
    - data_flat: Numpy array of saliency scores (of length 34716)
    - suffix: contain metadata to be included in filename (e.g. ABIDE_SEED_10_fold_0_t_1.0_layer_0) (str) 

    """

    plt.figure()
    bin_heights, bin_borders, _  = plt.hist(data_flat, bins = 200)
    plt.ylabel('Number of neurons')
    plt.xlabel(r'Salience score $c_{i}$, num = ' + str(len(data_flat)) + ', max = ' + str(np.max(data_flat)) + ', min = ' + str(np.min(data_flat))+  ', mean = ' + str(np.mean(data_flat)) +  ', std = ' + str(np.std(data_flat)))
    savefig(TARGET_DIRECTORY + 'importance_scores/' + 'importance_scores_hist_layer_' + str(suffix) + '.png', bbox_inches='tight', format='png', dpi=200) 

## Data Preparation

def prepare_dataset(class_subset, mask_initial):
    """
    Wrapper function to another dataset specific function for dataset creation

    Inputs:
    - class_subset: one of 'CN-AD' 'CN-MCI' 'MDD' 'ADHD' 'ABIDE' (str) 
    - mask_initial: Numpy array containing the existing mask, for repeated removal of features (not used here, so it is always a simple mask of all 1s)

    Returns:
    - subject_groups_: list of subject names, used for creating folds that ensure that a subject isn't found in both train and test set
    - X__: Numpy array of matrices containing the dataset
    - Y__: Numpy array containing the dataset labels
    - hidden_layer_size: number of nodes to be used for the neural network, for this dataset (determined from grid search done previously)
    - model_type: type of neural network to be used, for this dataset (determined from grid search done previously)
    """

    if class_subset == 'CN-MCI': 
        (subject_groups_, X__, Y__) = prepare_dataset_ADNI_matrices_masked(class_subset, mask_initial)
        hidden_layer_size = 20
        model_type = 'funcNetFFN_3L'

    elif class_subset == 'CN-AD': 
        (subject_groups_, X__, Y__) = prepare_dataset_ADNI_matrices_masked(class_subset, mask_initial)
        hidden_layer_size = 50
        model_type = 'funcNetFFN_3L'

    elif class_subset == 'MDD':
        (subject_groups_, X__, Y__) = prepare_dataset_MDD_matrices_masked(mask_initial)
        hidden_layer_size = 1000
        model_type = 'funcNetFFN_3L'

    elif class_subset == 'ADHD':
        (subject_groups_, X__, Y__) = prepare_dataset_ADHD_matrices_masked(mask_initial)
        hidden_layer_size = 50
        model_type = 'funcNetFFN_3L'

    elif class_subset == 'ABIDE':
        (subject_groups_, X__, Y__) = prepare_dataset_ABIDE_matrices_masked(mask_initial)
        hidden_layer_size = 50
        model_type = 'funcNetFFN_2L'
 
    return (subject_groups_, X__, Y__, hidden_layer_size, model_type)

def prepare_dataset_ADNI_matrices_masked(choice, mask):
    """
    Code to prepare the ADNI dataset
    Reads in .npy files from subfolders (for each class), combine into a list/numpy array and returns them

    Inputs:
    - choice: one of 'CN-AD' 'CN-MCI' (str) 
    - mask: Numpy array containing the existing mask, for repeated removal of features (not used here, so it is always a simple mask of all 1s)

    Returns:
    - subject_names_list: list of subject names, used for creating folds that ensure that a subject isn't found in both train and test set
    - all_matrices: Numpy array of matrices containing the dataset
    - Y: Numpy array containing the dataset labels
    """

    src_dir = '../data/ADNI/'

    if not (choice == 'CN-MCI' or choice == 'MCI-AD' or choice == 'CN-AD'):
        print('Invalid input detected. Allowable options: CN-MCI, MCI-AD, CN-AD')
        exit()

    subject_names_list = [] 
    num_remaining_features = np.count_nonzero(np.sum(mask, axis = 0), axis=None) 
    num_features = (num_remaining_features, num_remaining_features)
    
    non_zero_rows = np.where(np.sum(mask, axis = 0) > 0)[0]

    if 'CN' in choice:
        
        print('Preparing CN...')
        all_matrices_cn = []

        for i, file_or_dir in enumerate(os.listdir(src_dir + "CN/")):
            if ".DS_Store" not in file_or_dir:
                all_matrices_cn.append(np.load(src_dir + "CN/" + file_or_dir))
                subject_names_list.append(file_or_dir[10:18])

        for i, matrix in enumerate(all_matrices_cn):
            matrix = np.nan_to_num(matrix)
            masked_matrix = np.multiply(matrix, mask)
            reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
            all_matrices_cn[i] = reduced_matrix

    if 'MCI' in choice:
        
        print('Preparing MCI...')
        all_matrices_mci = []

        for i, file_or_dir in enumerate(os.listdir(src_dir + "MCI/")):
            if ".DS_Store" not in file_or_dir:
                all_matrices_mci.append(np.load(src_dir + "MCI/" + file_or_dir))
                subject_names_list.append(file_or_dir[10:18])

        for i, matrix in enumerate(all_matrices_mci):
            matrix = np.nan_to_num(matrix)
            masked_matrix = np.multiply(matrix, mask)
            reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
            all_matrices_mci[i] = reduced_matrix

    if 'AD' in choice:
        
        print('Preparing AD...')
        all_matrices_ad = []

        for i, file_or_dir in enumerate(os.listdir(src_dir + "AD/")):
            if ".DS_Store" not in file_or_dir:
                all_matrices_ad.append(np.load(src_dir + "AD/" + file_or_dir))
                subject_names_list.append(file_or_dir[10:18])

        for i, matrix in enumerate(all_matrices_ad):
            matrix = np.nan_to_num(matrix)
            masked_matrix = np.multiply(matrix, mask)
            reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
            all_matrices_ad[i] = reduced_matrix

    ## Combine

    if choice == 'CN-MCI':

        all_matrices = np.empty((len(all_matrices_cn) + len(all_matrices_mci), num_features[0], num_features[1]))

        for i, matrix in enumerate(all_matrices):  
            if i < len(os.listdir(src_dir + 'CN')): 
                all_matrices[i] = all_matrices_cn[i]
            elif i < len(os.listdir(src_dir + 'CN')) + len(os.listdir(src_dir + 'MCI')):
                all_matrices[i] = all_matrices_mci[i - (len(os.listdir(src_dir + 'CN')))]
            else: 
                print("There are more matrices than expected!")

        label_cn = [0 for i in range(len(all_matrices_cn))]
        label_mci = [1 for i in range(len(all_matrices_mci))]

        all_labels = np.array(label_cn + label_mci) 

        Y = np.zeros((all_matrices.shape[0], 2))
        for i in range(all_labels.shape[0]):
            Y[i, all_labels[i]] = 1 # 1-hot vectors

    elif choice == 'MCI-AD':
        all_matrices = np.empty((len(all_matrices_mci) + len(all_matrices_ad), num_features[0], num_features[1])) 

        for i, matrix in enumerate(all_matrices):  
            if i < len(os.listdir(src_dir + 'MCI')): 
                all_matrices[i] = all_matrices_mci[i]
            elif i < len(os.listdir(src_dir + 'MCI')) + len(os.listdir(src_dir + 'AD')):
                all_matrices[i] = all_matrices_ad[i - (len(os.listdir(src_dir + 'MCI')))]
            else: 
                print("There are more matrices than expected!")

        label_mci = [0 for i in range(len(all_matrices_mci))]
        label_ad = [1 for i in range(len(all_matrices_ad))]

        all_labels = np.array(label_mci + label_ad)

        Y = np.zeros((all_matrices.shape[0], 2))
        for i in range(all_labels.shape[0]):
            Y[i, all_labels[i]] = 1 # 1-hot vectors

    elif choice == 'CN-AD':
        all_matrices = np.empty((len(all_matrices_cn) + len(all_matrices_ad), num_features[0], num_features[1]))

        for i, matrix in enumerate(all_matrices):  
            if i < len(os.listdir(src_dir + 'CN')): 
                all_matrices[i] = all_matrices_cn[i]
            elif i < len(os.listdir(src_dir + 'CN')) + len(os.listdir(src_dir + 'AD')):
                all_matrices[i] = all_matrices_ad[i - (len(os.listdir(src_dir + 'CN')))]
            else: 
                print("There are more matrices than expected!")

        label_cn = [0 for i in range(len(all_matrices_cn))]
        label_ad = [1 for i in range(len(all_matrices_ad))]

        all_labels = np.array(label_cn + label_ad) 

        Y = np.zeros((all_matrices.shape[0], 2))
        for i in range(all_labels.shape[0]):
            Y[i, all_labels[i]] = 1 # 1-hot vectors
        
    else:
        print('Not possible to reach here!')
        exit()

    return (subject_names_list, all_matrices, Y)

def prepare_dataset_MDD_matrices_masked(mask):
    """
    Code to prepare the MDD dataset
    Reads in .npy files from subfolders (for each class), combine into a list/numpy array and returns them

    Inputs:
    - mask: Numpy array containing the existing mask, for repeated removal of features (not used here, so it is always a simple mask of all 1s)

    Returns:
    - subject_names_list: list of subject names, used for creating folds that ensure that a subject isn't found in both train and test set
    - all_matrices: Numpy array of matrices containing the dataset
    - Y: Numpy array containing the dataset labels
    """

    src_dir = '../data/MDD/'

    num_remaining_features = np.count_nonzero(np.sum(mask, axis = 0), axis=None) 
    num_features = (num_remaining_features, num_remaining_features)
    non_zero_rows = np.where(np.sum(mask, axis = 0) > 0)[0]

    all_matrices_normal = []
    subject_names_list = [] 

    for i, file_or_dir in enumerate(os.listdir(src_dir + "normal/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_normal.append(np.load(src_dir + "normal/" + file_or_dir))
            subject_names_list.append(file_or_dir[10:-4])

    for i, matrix in enumerate(all_matrices_normal):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_normal[i] = reduced_matrix

    all_matrices_diseased = []

    for i, file_or_dir in enumerate(os.listdir(src_dir + "diseased/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_diseased.append(np.load(src_dir + "diseased/" + file_or_dir))
            subject_names_list.append(file_or_dir[10:-4])

    for i, matrix in enumerate(all_matrices_diseased):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_diseased[i] = reduced_matrix

    all_matrices = np.empty((len(all_matrices_normal) + len(all_matrices_diseased), num_features[0], num_features[1]))

    for i, matrix in enumerate(all_matrices):  
        if i < len(os.listdir(src_dir + 'normal')): 
            all_matrices[i] = all_matrices_normal[i]
        elif i < len(os.listdir(src_dir + 'normal')) + len(os.listdir(src_dir + 'diseased')):
            all_matrices[i] = all_matrices_diseased[i - (len(os.listdir(src_dir + 'normal')))]
        else: 
            print("There are more matrices than expected!")

    label_normal = [0 for i in range(len(all_matrices_normal))]
    label_diseased = [1 for i in range(len(all_matrices_diseased))]

    all_labels = np.array(label_normal + label_diseased) 

    Y = np.zeros((all_matrices.shape[0], 2))
    for i in range(all_labels.shape[0]):
        Y[i, all_labels[i]] = 1 # 1-hot vectors

    return (subject_names_list, all_matrices, Y)

def prepare_dataset_ADHD_matrices_masked(mask):
    """
    Code to prepare the ADHD dataset
    Reads in .npy files from subfolders (for each class), combine into a list/numpy array and returns them

    Inputs:
    - mask: Numpy array containing the existing mask, for repeated removal of features (not used here, so it is always a simple mask of all 1s)

    Returns:
    - subject_names_list: list of subject names, used for creating folds that ensure that a subject isn't found in both train and test set
    - all_matrices: Numpy array of matrices containing the dataset
    - Y: Numpy array containing the dataset labels
    """

    src_dir = '../data/ADHD/'

    num_remaining_features = np.count_nonzero(np.sum(mask, axis = 0), axis=None) 
    num_features = (num_remaining_features, num_remaining_features)
    non_zero_rows = np.where(np.sum(mask, axis = 0) > 0)[0]

    all_matrices_normal = []
    subject_names_list = []

    for i, file_or_dir in enumerate(os.listdir(src_dir + "normal/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_normal.append(np.load(src_dir + "normal/" + file_or_dir))
            subject_names_list.append(file_or_dir[0:10])

    for i, matrix in enumerate(all_matrices_normal):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_normal[i] = reduced_matrix

    all_matrices_diseased = []

    for i, file_or_dir in enumerate(os.listdir(src_dir + "diseased/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_diseased.append(np.load(src_dir + "diseased/" + file_or_dir))
            subject_names_list.append(file_or_dir[0:10])

    for i, matrix in enumerate(all_matrices_diseased):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_diseased[i] = reduced_matrix

    all_matrices = np.empty((len(all_matrices_normal) + len(all_matrices_diseased), num_features[0], num_features[1]))

    for i, matrix in enumerate(all_matrices):  
        if i < len(os.listdir(src_dir + 'normal')): 
            all_matrices[i] = all_matrices_normal[i]
        elif i < len(os.listdir(src_dir + 'normal')) + len(os.listdir(src_dir + 'diseased')):
            all_matrices[i] = all_matrices_diseased[i - (len(os.listdir(src_dir + 'normal')))]
        else: 
            print("There are more matrices than expected!")

    label_normal = [0 for i in range(len(all_matrices_normal))]
    label_diseased = [1 for i in range(len(all_matrices_diseased))]

    all_labels = np.array(label_normal + label_diseased) 

    Y = np.zeros((all_matrices.shape[0], 2))
    for i in range(all_labels.shape[0]):
        Y[i, all_labels[i]] = 1 # 1-hot vectors

    return (subject_names_list, all_matrices, Y)

def prepare_dataset_ABIDE_matrices_masked(mask):
    """
    Code to prepare the ABIDE (ASD) dataset
    Reads in .npy files from subfolders (for each class), combine into a list/numpy array and returns them

    Inputs:
    - mask: Numpy array containing the existing mask, for repeated removal of features (not used here, so it is always a simple mask of all 1s)

    Returns:
    - subject_names_list: list of subject names, used for creating folds that ensure that a subject isn't found in both train and test set
    - all_matrices: Numpy array of matrices containing the dataset
    - Y: Numpy array containing the dataset labels
    """

    src_dir = '../data/ABIDE/'

    num_remaining_features = np.count_nonzero(np.sum(mask, axis = 0), axis=None) 
    num_features = (num_remaining_features, num_remaining_features)
    non_zero_rows = np.where(np.sum(mask, axis = 0) > 0)[0]

    all_matrices_normal = []
    subject_names_list = []

    for i, file_or_dir in enumerate(os.listdir(src_dir + "normal/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_normal.append(np.load(src_dir + "normal/" + file_or_dir))
            subject_names_list.append(file_or_dir[0:-10])

    for i, matrix in enumerate(all_matrices_normal):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_normal[i] = reduced_matrix

    all_matrices_diseased = []

    for i, file_or_dir in enumerate(os.listdir(src_dir + "diseased/")):
        if ".DS_Store" not in file_or_dir:
            all_matrices_diseased.append(np.load(src_dir + "diseased/" + file_or_dir))
            subject_names_list.append(file_or_dir[0:-10])

    for i, matrix in enumerate(all_matrices_diseased):
        matrix = np.nan_to_num(matrix)
        masked_matrix = np.multiply(matrix, mask)
        reduced_matrix = masked_matrix[np.ix_(non_zero_rows, non_zero_rows)]
        all_matrices_diseased[i] = reduced_matrix

    all_matrices = np.empty((len(all_matrices_normal) + len(all_matrices_diseased), num_features[0], num_features[1]))

    for i, matrix in enumerate(all_matrices):  
        if i < len(os.listdir(src_dir + 'normal')): 
            all_matrices[i] = all_matrices_normal[i]
        elif i < len(os.listdir(src_dir + 'normal')) + len(os.listdir(src_dir + 'diseased')):
            all_matrices[i] = all_matrices_diseased[i - (len(os.listdir(src_dir + 'normal')))]
        else: 
            print("There are more matrices than expected!")

    label_normal = [0 for i in range(len(all_matrices_normal))]
    label_diseased = [1 for i in range(len(all_matrices_diseased))]

    all_labels = np.array(label_normal + label_diseased) 

    Y = np.zeros((all_matrices.shape[0], 2))
    for i in range(all_labels.shape[0]):
        Y[i, all_labels[i]] = 1 # 1-hot vectors

    return (subject_names_list, all_matrices, Y)

def split_kfoldcv_sbj(Y, subjects, n, seed):
    """
    Performs a split on the dataset to allow k-fold cross valiation
    Ensures that a subject isn't found in both train and test set

    Inputs:
    - Y: Numpy array containing the dataset labels
    - subjects: list of subject names, used for creating folds that ensure that a subject isn't found in both train and test set
    - n: number of splits (int)  
    - seed: seed number (int)

    Returns:
    - result: List of tuples containing (Numpy array of indices in training set, Numpy array of indices in test set)
    """

    unique = np.unique(subjects)
    subject_Y = []

    for subject in unique:
        y = Y[subjects.index(subject)]
        subject_Y.append(y)

    subject_X = np.zeros_like(unique)
    skf_group = StratifiedKFold(n_splits = n, random_state = seed, shuffle=True)
    
    result = []
    for train_index, test_index in skf_group.split(subject_X, subject_Y):
        train_subjects_in_fold = unique[train_index]
        test_subjects_in_fold = unique[test_index]

        train = np.in1d(subjects, train_subjects_in_fold).nonzero()[0]
        test = np.in1d(subjects, test_subjects_in_fold).nonzero()[0]

        result.append((train, test))

    return result
 