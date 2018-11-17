import os
import re
import time

import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt

from eval_plot.evaluation import evaluate
from eval_plot.evaluation import ploting_v
from eval_plot.evaluation import ploting_v3d
from algorithms.kmeans import Kmeans
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

# -------------------------------------------------------------------------------------------------------------- K-means
def tester_kmeans(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 2        # Number of clusters
    num_tries_init = 1      # Number of different initializations of the centroids
    max_iterations = 6     # Number of iterations for each initialization

    print('\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m'+'\nNumber of clusters: '+str(
        num_clusters)+'\nNumber of different initilizations: '+str(num_tries_init)+'\nMaximum number of iterations '
                                                                            'per initialization: '+str(max_iterations))

    start_time = time.time()
    tst2 = Kmeans(num_clusters, num_tries_init, max_iterations)
    tst2.kmeans_method(data_x)
    print('Running time: %s seconds' % round(time.time() - start_time, 4))
    evaluate(tst2.labels_km, groundtruth_labels, data_x)
    return tst2.labels_km

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

    # -------------------------------------------------------------------------------Compute covariance and eigenvectors
    original_mean = np.mean(data_x, axis=0)
    # data_x = data_x-original_mean  # Substracting the mean of the data
    # original_mean = np.mean(data_x, axis=0)  # Recomputing the mean (checkpoint to check if it is 0 for each feature)

    cov_m = compute_covariance(data_x, original_mean)
    eig_vals, eig_vect = np.linalg.eig(cov_m)

    idxsort = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idxsort]
    eig_vect = eig_vect[:,idxsort]

    # ---------------------------------------------------------------------Decide the number of features we want to keep
    prop_variance = 0.9
    k = proportion_of_variance(eig_vals, prop_variance)
    print('\nThe value of K selected to obtain a proportion of variance = '+str(prop_variance) +' is: ' + str(k)+'\n')

    eig_vals_red = eig_vals[:k]
    eig_vect_red = eig_vect[:,:k]  # Eigenvectors are in columns (8xk)

    # ---------------------------------------------------------------------------------Reduce dimensionality of the data
    # A1) Using our implementation of PCA
    transf_data_x = np.dot((eig_vect_red.T), (data_x-original_mean).T).T

    # B1) Using the PCA implementation of sklearn
    pca = PCA(n_components=k)
    transf_data_x_sklearn = pca.fit_transform(data_x)

    # C1) Using the incremental PCA implementation of sklearn
    incrementalpca = IncrementalPCA(n_components=k)
    transf_data_x_sklearn2 = incrementalpca.fit_transform(data_x)

    # --------------------------------------------------------------------------------------------------Reconstruct data
    # A2) Reconstruct data with our method
    reconstruct_data_x = np.dot(eig_vect_red, transf_data_x.T)
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
    print('The relative error after reconstructing the original matrix with K = ' + str(k) + ' is '+'\033[1m'+'\033['
    '94m'+str(round(total_error,2)) + '%' +'\033[0m'+' [using our implementation of PCA]')
    # identity_aproximation = np.dot(eig_vect, eig_vect.T)

    # B3) Error between original data and reconstruct data 1
    error1 = reconstruct_data_x1-data_x
    total_error1 = (np.sum(abs(error1))/np.sum(abs(data_x)))*100
    print('The relative error after reconstructing the original matrix with K = ' + str(k) + ' is '+'\033[1m'+'\033['
    '94m'+str(round(total_error1,2)) +'%' +'\033[0m'+' [using pca.fit_transform of Sklearn]')

    # C3) Error between original data and reconstruct data 2
    error2 = reconstruct_data_x2-data_x
    total_error2 = (np.sum(abs(error2))/np.sum(abs(data_x)))*100
    print('The relative error after reconstructing the original matrix with K = ' + str(k) + ' is '+'\033[1m'+'\033['
    '94m'+str(round(total_error2,2)) + '%' +'\033[0m'+' [using incrementalpca.fit_transform of Sklearn]')

    # ------------------------------------------------------------------------------Kmeans with dimensionality reduction
    print('\n---------------------------------------------------------------------------------------------------------')
    print('K-MEANS APPLIED TO THE ORIGINAL DATA')
    tester_kmeans(data_x, groundtruth_labels)
    # We could renormalize the data again to improve the performance of the Kmeans algorithm
    print('\n---------------------------------------------------------------------------------------------------------')
    print('K-MEANS APPLIED TO THE TRANSFORMED DATA USING OUR IMPLEMENTATION OF PCA')
    labels = tester_kmeans(transf_data_x, groundtruth_labels)
    print('\n---------------------------------------------------------------------------------------------------------')
    print('K-MEANS APPLIED TO THE TRANSFORMED DATA USING pca.fit_transform OF SKLEARN')
    tester_kmeans(transf_data_x_sklearn, groundtruth_labels)
    print('\n---------------------------------------------------------------------------------------------------------')
    print('K-MEANS APPLIED TO THE TRANSFORMED DATA USING incrementalpca.fit_transform OF SKLEARN')
    tester_kmeans(transf_data_x_sklearn2, groundtruth_labels)
    print('\n---------------------------------------------------------------------------------------------------------')

    # -----------------------------------------------------------------------------------------------------Scatter plots
    ploting_boolean = True

    if ploting_boolean:
        # Plot eigenvector
        plt.plot(eig_vals, 'ro-', linewidth=2, markersize=6)
        plt.title('Magnitude of the eigenvalues')
        plt.show()

        # Plottings: scatter plots
        # Original data with groundtruth labels
        ploting_v(data_x, num_clusters, groundtruth_labels)
        # Transfomed data with our implementation of PCA and with groundtruth labels
        ploting_v(transf_data_x, num_clusters, groundtruth_labels)
        # Transfomed data with pca.fit_transform and with groundtruth labels
        ploting_v(transf_data_x_sklearn, num_clusters, groundtruth_labels)
        # Transfomed data with incrementalpca.fit_transform and with groundtruth labels
        ploting_v(transf_data_x_sklearn2, num_clusters, groundtruth_labels)

        # ------------------------------------------------------------------------------------------------------3D plots
        # Plottings: 3D plots
        # Original data without labels
        ploting_v3d(data_x, 1, np.zeros(len(groundtruth_labels)), 'original data without labels')
        # Original data with groundtruth labels
        ploting_v3d(data_x, num_clusters, groundtruth_labels, 'original data with groundtruth labels')
        # Reconstructed data without labels
        ploting_v3d(reconstruct_data_x, 1, np.zeros(len(groundtruth_labels)), 'reconstructed data without labels')
        # Transfomed data with our implementation of PCA and without labels
        ploting_v3d(transf_data_x, 1, np.zeros(len(groundtruth_labels)), 'transformed data without labels')
        # Transfomed data with our implementation of PCA and with groundtruth_labels
        ploting_v3d(transf_data_x, num_clusters, groundtruth_labels, 'transformed data with groundtruth labels')
        # Transfomed data with our implementation of PCA and with the labels obtained with our K-means
        ploting_v3d(transf_data_x, num_clusters, labels, 'transformed data with labels from our K-means')


# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()
