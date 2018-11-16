import os
import re
import time

import pandas as pd
import numpy as np
from scipy.io import arff

from eval_plot.evaluation import evaluate
from eval_plot.evaluation import ploting_v
from eval_plot.evaluation import ploting_v3d
from matplotlib import pyplot
from algorithms.methods import compute_covariance
from algorithms.methods import proportion_of_variance
from preproc.preprocess import Preprocess
from sklearn.preprocessing.label import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA


# ------------------------------------------------------------------------------------------------------- Read databases
def obtain_arffs(path):
    # Read all the databases
    arffs_dic = {}
    for filename in os.listdir(path):
        if re.match('(.*).arff', filename):
            arffs_dic[filename.replace('.arff', '')] = arff.loadarff(path + filename)
    return arffs_dic

# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the datasets...' + '\033[0m')
    arffs_dic = obtain_arffs('./datasets/')

    # Extract an specific database
    dataset_name = 'breast-w'       # possible datasets ('hypothyroid', 'breast-w', 'waveform')
    dat1 = arffs_dic[dataset_name]
    df1 = pd.DataFrame(dat1[0])     # original data in pandas dataframe
    groundtruth_labels = df1[df1.columns[len(df1.columns)-1]].values  # original labels in a numpy array
    df1 = df1.drop(df1.columns[len(df1.columns)-1],1)
    if dataset_name == 'hypothyroid':
        df1 = df1.drop('TBG', 1)    # This column only contains NaNs so does not add any value to the clustering
    data1 = df1.values              # original data in a numpy array without labels
    load = Preprocess()
    data_x = load.preprocess_method(data1)
    le = LabelEncoder()
    le.fit(np.unique(groundtruth_labels))
    groundtruth_labels = le.transform(groundtruth_labels)

    num_clusters = len(np.unique(groundtruth_labels)) # Number of different labels

    original_mean = np.mean(data_x, axis=0)
    cov_m = compute_covariance(data_x, original_mean)
    eig_vals, eig_vect = np.linalg.eig(cov_m)

    idxSort = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idxSort]
    eig_vect =eig_vect[:,idxSort]

    k = proportion_of_variance(eig_vals, 0.9)

    eig_vals = eig_vals[:k]
    eig_vect = eig_vect[:,:k] # Eigenvectors are in columns (8xk)

    # ---------------------------------------------------------------------------------Reduce dimensionality of the data
    # A1) Using our implementation of PCA
    transf_data_x = np.dot((eig_vect.T), (data_x-original_mean).T).T

    # B1) Using the PCA implementation of sklearn
    pca = PCA(n_components=k)
    transf_data_x_sklearn = pca.fit_transform(data_x)

    # C1) Using the incremental PCA implementation of sklearn
    incrementalpca = IncrementalPCA(n_components=k)
    transf_data_x_sklearn2 = incrementalpca.fit_transform(data_x)

    # --------------------------------------------------------------------------------------------------Reconstruct data
    # A2) Reconstruct data with our method
    reconstruct_data_x = np.dot(eig_vect, transf_data_x.T)
    reconstruct_data_x = reconstruct_data_x.T + original_mean

    # B2) Reconstruct data with PCA sklearn
    reconstruct_data_x1 = np.dot(pca.components_.T, transf_data_x_sklearn.T)
    reconstruct_data_x1 = reconstruct_data_x1.T + original_mean

    # C2) Reconstruct data with incremental PCA sklearn
    reconstruct_data_x2 = np.dot(incrementalpca.components_.T, transf_data_x_sklearn2.T)
    reconstruct_data_x2 = reconstruct_data_x2.T + original_mean

    # ----------------------------------------------------------------Error between original data and reconstructed data
    # A3) Error between original data and reconstruct data
    error = reconstruct_data_x-data_x
    total_error = (np.sum(abs(error))/np.sum(abs(data_x)))*100
    print('The total error after reconstructing the original matrix with K = ' + str(k) + ' is '+str(
        round(total_error,2)) + '%')
    # identity_aproximation = np.dot(eig_vect, eig_vect.T)

    # B3) Error between original data and reconstruct data 1
    error1 = reconstruct_data_x1-data_x
    total_error1 = (np.sum(abs(error1))/np.sum(abs(data_x)))*100
    print('The total error after reconstructing the original matrix with K = ' + str(k) + ' is '+str(
        round(total_error1,2)) + '%')

    # C3) Error between original data and reconstruct data 2
    error2 = reconstruct_data_x2-data_x
    total_error2 = (np.sum(abs(error2))/np.sum(abs(data_x)))*100
    print('The total error after reconstructing the original matrix with K = ' + str(k) + ' is '+str(
        round(total_error2,2)) + '%')

    # -----------------------------------------------------------------------------------------------------Scatter plots
    # Plottings: scatter plots
    ploting_v(data_x, num_clusters, groundtruth_labels) # Original data
    ploting_v(transf_data_x, num_clusters, groundtruth_labels) # Using our implementation of PCA
    ploting_v(transf_data_x_sklearn, num_clusters, groundtruth_labels) # Using the PCA implementation of sklearn
    ploting_v(transf_data_x_sklearn2, num_clusters, groundtruth_labels) # Using the incremenatl PCA implementation of sk


    # ----------------------------------------------------------------------------------------------------------3D plots
    # Plottings: 3D plots
    ploting_v3d(transf_data_x, num_clusters, groundtruth_labels) # Transfomed data with groundtruth_labels
    # ploting_v3d(transf_data_x, num_clusters, labels) # Transfomed data with the labels obtained with our kmeans


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()
