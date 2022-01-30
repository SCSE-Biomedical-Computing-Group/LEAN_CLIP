# Obtaining leaner DNN for decoding brain functional connectome in a single shot

In this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231221000977), we have proposed 2 algorithms: Layerwise Elimination of Accessory Nodes (LEAN) and Correlation-based eLimination of InPuts (CLIP).

This code repository contains the implementation of 3 configurations as discussed in the paper:
1. LEAN 
2. LEAN (Inputs only) + CLIP
3. LEAN + CLIP

## Setup

`pip3 install -r requirements.txt`

## Guide to run the code

1. Download the [ABIDE dataset](https://ida.loni.usc.edu/login.jsp?project=ABIDE) or [ADHD dataset](http://fcon_1000.projects.nitrc.org/indi/adhd200/) or [ADNI dataset](https://ida.loni.usc.edu/login.jsp?project=ADNI). You will need to request for access if you do not have an account with LONI's IDA / NITRC. 

   Following the data processing steps as detailed in the paper, the connectivity matrices should be saved with the following filenames and directory structure. This is so as the subject IDs are extracted from the filenames when performing k-fold splits, so as to ensure that the same subject is not found in both the training and test set. 
   
   Alternatively, the `prepare_dataset_<dataset_name>_matrices_masked` functions in `utils.py` can be modified to adapt to any new filename or directory configuration. 

   - ABIDE: NYU_<subject_id>_power.npy
      - /data/ABIDE/diseased/
      - /data/ABIDE/normal/
   - ADHD: NYU-<subject_id>\_session\_<sess_id>\_rest\_<run_id>_power.npy
      - /data/ADHD/diseased/
      - /data/ADHD/normal/
   - CN-AD: Power_sub-<sub_id>_ses-<session_id>.npy
   - CN-MCI: Power_sub-<sub_id>_ses-<session_id>.npy
      - /data/ADNI/AD/
      - /data/ADNI/CN/
      - /data/ADNI/MCI/
   - MDD: Power_patient_<subject_id>_.npy
      - /data/MDD/diseased/
      - /data/MDD/normal/

   The exact filenames might vary for ABIDE and ADHD, e.g. 'NYU' might be replaced with another institution name.

   For CN-AD and CN-MCI, some sessions can have multiple runs, e.g. Power_sub-<sub_id>_ses-<session_id>_run-0.npy. As long as the core (prefix) structure follows the above format, the code will run.

   

2. To test CLIP, you will need to generate cluster_mask - an array with 0s and 1s to be applied on the input features. You can do so with the following code snippet:
   
   `python3 generate_CLIP_features.py ABIDE`

   You should modify the MULTIPROC_BATCH_SIZE parameter according to how much memory you have available in your system.

   To test other datasets, replace 'ABIDE' with the names of the other datasets.

3. To run the configurations, you will just need to run `main.py`. You can do so with the following code snippet:

   `python3 main.py ABIDE 0`

   - Possible options: 'CN-AD' 'CN-MCI' 'MDD' 'ADHD' 'ABIDE' 
   - 0 represents the gpu_id to use. This can be modified in `main.py`.

## Troubleshooting

If you're using a Mac, you might run into errors (e.g. IndexError: list index out of range) when reading in the data due to the presence of .DS_Store files. To fix it, in your data folder, execute the command below to remove them. 

`find . -name ".DS_Store" -delete -print`
