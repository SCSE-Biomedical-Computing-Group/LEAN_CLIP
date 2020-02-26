"""
Code to generate CLIP features

1. Generate correlation matrix 
2. Generate eigenvalue of correlation matrix, find elbow point and compute clusters
3. Generate cluster_mask, to be applied on the array of features

"""

import os
import sys

import h5py
from multiprocessing import Pool 

from kneed import KneeLocator
import numpy as np
import scipy 
from sklearn import cluster

from utils import corr_mx_flatten, get_keep_indices_from_mapping, map_main_to_sub, mkdir, prepare_dataset, split_kfoldcv_sbj


SPARSE_THRESHOLD_CORR = 0.3
CLUSTER_KEEP_RATIO = 0.05

MULTIPROC_BATCH_SIZE = 10 # each additional process takes ~10GB of memory

folds = 5 
seeds = list(range(10, 20))

mask_initial = np.ones((264, 264))

class_subset = str(sys.argv[1]) # 'CN-AD' 'CN-MCI' 'MDD' 'ADHD' 'ABIDE' 


def generate_corr_matrix(X__, seeds, folds):
    """
    Generate a correlation matrix from the given dataset
    All seeds and folds are done in this function (multiprocessing takes up too much memory)
    Only the training set is used to generate the matrices
    
    Inputs:
    - X__: Numpy array of matrices containing the dataset (training set)
    - seeds: list of seed numbers to use 
    - folds: number of folds (int)
    """
    
    TARGET_DIRECTORY = '../data/corr_matrix/' + class_subset + '/'
    mkdir(TARGET_DIRECTORY)
    
    for SEED in seeds:
    
        np.random.seed(SEED)
        idx = np.arange(len(X__))
        np.random.shuffle(idx) # randomize index
        
        X, Y, subject_groups = X__[idx], Y__[idx],  np.array(subject_groups_)[idx] 
        subject_groups = subject_groups.tolist()
        
        folds_indices = split_kfoldcv_sbj(Y.argmax(1), subject_groups, folds, SEED)

        fold_count = 0

        for train_index, val_index in folds_indices: # for each fold

            if os.path.exists(TARGET_DIRECTORY + "corr_matrix_seed" + str(SEED) + "_fold_" + str(fold_count) + ".hdf5"):
                print("corr_matrix_seed" + str(SEED) + "_fold_" + str(fold_count) + " has already been generated, skipping it...")

            else:
                print("corr_matrix_seed" + str(SEED) + "_fold_" + str(fold_count) + ".hdf5 not found!")

                X_ = corr_mx_flatten(X)

                X_train, Y_train = X_[train_index], Y[train_index]
                X_val, Y_val = X_[val_index], Y[val_index]
                
                corr_matrix = np.corrcoef(X_train.T) # Generate correlation matrix
                print('Correlation matrix generated for seed ' + str(SEED) + ' fold ' + str(fold_count))

                corr_matrix = np.absolute(corr_matrix)
                corr_matrix[corr_matrix < SPARSE_THRESHOLD_CORR] = 0
                print("Number of non-zero elements in corr_matrix: " + str(np.count_nonzero(corr_matrix)))

                g = h5py.File(TARGET_DIRECTORY + "corr_matrix_seed" + str(SEED) + "_fold_" + str(fold_count) + ".hdf5", "w")
                g.create_dataset('corr_matrix', data=corr_matrix)
                g.close()
                print("Wrote corr_matrix " + "corr_matrix_seed" + str(SEED) + "_fold_" + str(fold_count) + " to " + TARGET_DIRECTORY)

            fold_count += 1

def generate_eigenvalues_elbow_and_clusters(X__, fold_number, seed_number, min_labels=0, max_labels=1000):
    """
    Generates eigenvalues, the elbow point found by the Kneed library and the cluster number assigned to each datapoint
    Uses multiprocessing

    Inputs:
    - X__: Numpy array of matrices from the training set
    - fold_number (int)
    - seed_number (int)
    - min_labels: Used for finding the elbow
    - max_labels: Used for finding the elbow (must be same as param k in scipy.sparse.linalg.eigs)

    Returns:
    - eigenvalues: Numpy array of eigenvalues, of length k
    - kn.knee: knee point found (int)
    - group_labels_gmm: Numpy array showing the cluster number each data point is assigned to
    - seed_number (int)
    - fold_number (int)
    """
    
    print('Generating correlation matrix...') # this is faster than loading it since we're using multiprocessing
    corr_matrix = np.corrcoef(X__.T)
    corr_matrix = np.absolute(corr_matrix)
    corr_matrix[corr_matrix < SPARSE_THRESHOLD_CORR] = 0 
    print('Correlation matrix generated for seed ' + str(seed_number) + ' fold ' + str(fold_number))

    eigenvalues, _ = scipy.sparse.linalg.eigs(corr_matrix, k=max_labels) # reduce k to reduce time taken
    eigenvalues = np.real(eigenvalues) # drop complex numbers
    eigenvalues = np.absolute(eigenvalues)

    N = max_labels + 1
    ind = np.arange(1, N, 1) # the x locations for the groups
    kn = KneeLocator(ind, eigenvalues[min_labels:N], S=1.0, curve='convex', direction='decreasing', interp_method='polynomial')
    print("Elbow for seed " + str(seed_number) + " fold " + str(fold_number) + ": " + str(kn.knee))

    print("performing spectral clustering...")
    group_labels_gmm = cluster.SpectralClustering(n_clusters= kn.knee, random_state = int(seed_number), n_init = 1000, affinity='precomputed', assign_labels='discretize').fit_predict(corr_matrix)
    print("Clusters generated for seed " + str(seed_number) + " fold " + str(fold_number))

    return (eigenvalues, kn.knee, group_labels_gmm, seed_number, fold_number)

def generate_cluster_masks(class_subset, seeds, folds):
    """
    Generates a length 34716 array that can be applied to remove features
    Requires correlation matrices and clusters to be generated beforehand by generate_corr_matrix and generate_eigenvalues_elbow_and_clusters
    Creates a cluster_mask CSV file that will be used by other files later

    Inputs: 
    - class_subset: one of 'CN-AD' 'CN-MCI' 'MDD' 'ADHD' 'ABIDE' (str) 
    - seeds: list of seed numbers
    - folds: number of folds (int)
    """

    SOURCE_DIRECTORY = '../data/eigenvalues/' + class_subset + '/'
    SOURCE_DIRECTORY_CORR = '../data/corr_matrix/' + class_subset + '/'
    TARGET_DIRECTORY = '../data/cluster_mask/' + class_subset + '/'

    mkdir(TARGET_DIRECTORY)

    for SEED in seeds:
        for fold_count in range(folds):

            if os.path.exists(TARGET_DIRECTORY + class_subset + "_cluster_mask_seed" + str(SEED) + "_fold_" + str(fold_count) + ".csv"):
                print(class_subset + "_cluster_mask_seed" + str(SEED) + "_fold_" + str(fold_count) + " has already been generated, skipping it...")
                continue

            group_labels_gmm = np.loadtxt(SOURCE_DIRECTORY + "clusters_seed" + str(SEED) + "_fold_" + str(fold_count) + ".csv").astype(int)

            f1 = h5py.File(SOURCE_DIRECTORY_CORR + 'corr_matrix_seed' + str(SEED) + '_fold_' + str(fold_count) + '.hdf5', 'r')
            corr_matrix = f1.get('corr_matrix').value 
            print('Read corr_matrix for seed ' + str(SEED) + ' fold ' + str(fold_count))

            # Mask generation
            cluster_ids, num_cluster_elements = np.unique(group_labels_gmm, return_counts = True)
            keep_indices = np.array([])
            print("Number of clusters found: " + str(len(cluster_ids)) + ' for seed ' + str(SEED) + ' fold ' + str(fold_count))

            for cluster_id, cluster_id_size in zip(cluster_ids, num_cluster_elements):

                if cluster_id_size * 0.01 < 1: # less than 100 elements
                    num_elements_to_keep = int(CLUSTER_KEEP_RATIO*100)
                    if num_elements_to_keep > cluster_id_size:
                        num_elements_to_keep = int(cluster_id_size)
                else: # > 100 elements, take x%
                    num_elements_to_keep = round(CLUSTER_KEEP_RATIO * cluster_id_size).astype(int)

                cluster_id_indices = np.where(group_labels_gmm == cluster_id)[0]
                mapping = map_main_to_sub(cluster_id_indices)
                sub_corr_matrix = corr_matrix[np.ix_(cluster_id_indices, cluster_id_indices)]
                sub_corr_matrix = np.absolute(sub_corr_matrix)
                node_corr_scores = np.sum(sub_corr_matrix, axis = 1)
                node_corr_rank = np.argsort(-node_corr_scores) # descending

                chosen_new_indices = node_corr_rank[:num_elements_to_keep]
                chosen_indices = get_keep_indices_from_mapping(mapping, chosen_new_indices)
                keep_indices = np.concatenate((keep_indices, chosen_indices), axis = None).astype(int)

            cluster_mask = np.zeros(len(group_labels_gmm))
            cluster_mask[keep_indices] = 1
            cluster_mask = cluster_mask.astype(int)

            np.savetxt(TARGET_DIRECTORY + class_subset + '_cluster_mask_seed' + str(SEED) + '_fold_' + str(fold_count) + '.csv', np.transpose(cluster_mask), delimiter= ',')
            print("Wrote " + class_subset + "_cluster_mask_seed" + str(SEED) + "_fold_" + str(fold_count) + " to " + TARGET_DIRECTORY)

### Main code begins

(subject_groups_, X__, Y__, hidden_layer_size, cnn_model) = prepare_dataset(class_subset, mask_initial)

## Correlation matrices

generate_corr_matrix(X__, seeds, folds)

## Eigenvalues and Clusters

TARGET_DIRECTORY = '../data/eigenvalues/' + class_subset + '/'
mkdir(TARGET_DIRECTORY)

list_of_multiproc_input = []

for SEED in seeds:

    np.random.seed(SEED)
    idx = np.arange(len(X__))
    np.random.shuffle(idx) # randomize index
    
    X, Y, subject_groups = X__[idx], Y__[idx], np.array(subject_groups_)[idx] 
    subject_groups = subject_groups.tolist()
    
    folds_indices = split_kfoldcv_sbj(Y.argmax(1), subject_groups, folds, SEED)

    fold_count = 0

    for train_index, val_index in folds_indices: # for each fold

        if os.path.exists(TARGET_DIRECTORY + "eigv_seed" + str(SEED) + "_fold_" + str(fold_count) + ".csv"):
            print("eigv_seed" + str(SEED) + "_fold_" + str(fold_count) + " has already been generated, skipping it...")

        else:
            print("eigv_seed" + str(SEED) + "_fold_" + str(fold_count) + ".csv not found!")

            X_ = corr_mx_flatten(X)

            X_train, Y_train = X_[train_index], Y[train_index]
            X_val, Y_val = X_[val_index], Y[val_index]

            list_of_multiproc_input.append((X_train, fold_count, SEED))

        fold_count += 1
 
with Pool(processes=MULTIPROC_BATCH_SIZE) as pool:
    result = pool.starmap_async(generate_eigenvalues_elbow_and_clusters, list_of_multiproc_input)   
    result = result.get()

for eigv, elbow, clusters, seed_number, fold_number in result:

    no_of_clusters = len(set(clusters))

    from collections import Counter
    cluster_names = np.fromiter(Counter(clusters).keys(),dtype=int)
    number_of_elements_in_each_cluster = np.fromiter(Counter(clusters).values(),dtype=int)
    inds = cluster_names.argsort()
    number_of_elements_in_each_cluster_sorted = number_of_elements_in_each_cluster[inds]
    cluster_names_sorted = cluster_names[inds]

    np.savetxt(TARGET_DIRECTORY + "eigv_seed" + str(seed_number) + "_fold_" + str(fold_number) + ".csv", eigv, delimiter=",")
    print("Wrote matrix eigvals " + "eigv_seed" + str(seed_number) + "_fold_" + str(fold_number) + " to " + TARGET_DIRECTORY)
    
    with open(TARGET_DIRECTORY + "clusters_elbow_seed_" + str(seed_number) + "_fold_" + str(fold_number) + ".csv", 'a+') as out_stream:
        out_stream.write(str(elbow) + ', ' + str(cluster_names_sorted) + ', ' + str(number_of_elements_in_each_cluster_sorted) + ', ' + str(clusters) + '\n')
    print("Wrote clusters and elbows " + "clusters_elbow_seed_" + str(seed_number) + "_fold_" + str(fold_number) + " to " + TARGET_DIRECTORY)

    np.savetxt(TARGET_DIRECTORY + "clusters_seed" + str(seed_number) + "_fold_" + str(fold_number) + ".csv", clusters, delimiter=",")
    print("Wrote clusters " + "clusters_seed_" + str(seed_number) + "_fold_" + str(fold_number) + " to " + TARGET_DIRECTORY)

## Cluster masks

generate_cluster_masks(class_subset, seeds, folds)
