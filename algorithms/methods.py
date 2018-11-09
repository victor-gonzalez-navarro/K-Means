import numpy as np

def compute_covariance(data_x, mean):
    data_E = data_x - mean

    cov_m = np.dot(data_E.T, data_E) / data_x.shape[0]

    return cov_m

def proportion_of_variance(eig_vals, prop):
    dim = sum((np.cumsum(eig_vals) / sum(eig_vals)) < prop) + 1
    return dim