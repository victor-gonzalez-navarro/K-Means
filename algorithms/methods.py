import numpy as np

def compute_covariance(data_x):
    mean_x = np.mean(data_x, axis=0)

    cov_m = np.zeros((data_x.shape[1],data_x.shape[1]))
    for i in range(data_x.shape[1]):
        for j in range(data_x.shape[1]):
            cov_m[i,j] = sum() / data_x.shape[0]