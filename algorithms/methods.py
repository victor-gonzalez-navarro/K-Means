import numpy as np

def compute_covariance(data_x):
    mean_x = np.mean(data_x, axis=0)

    cov_m = np.zeros((data_x.shape[1],data_x.shape[1]))
    for i in range(data_x.shape[1]):
        for j in range(data_x.shape[1]):
            cov_m[i,j] = np.dot((data_x[:,i]-mean_x[i]).T,(data_x[:,j]-mean_x[j])) / data_x.shape[0]

    return cov_m

def proportion_of_variance(eig_vals, prop):
    dim = sum((np.cumsum(eig_vals) / sum(eig_vals)) < prop) + 1
    return dim