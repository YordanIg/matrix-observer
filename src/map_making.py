"""
Reversing the forward-modelling process using maximum-likelihood methods.
"""
import numpy as np

def calc_ml_estimator_matrix(mat_A, mat_N, cov=False):
    """
    For the general problem 
        d = Aa + n
    where d is data, a is the alm vector, n is noise and A is the observation 
    matrix, calculate the matrix W = [ A^{T} N^{-1} A ]^{-1} A^{T} N^{-1}. This 
    allows the generalised least-squares solution
        \hat{a}_{ml} = W d
    assuming that noise is zero-mean and that N = <nn^{T}> (the noise 
    covariance).

    If cov is True, also returns the covariance matrix of \hat{a}_{ml} 
    (the "map"), given by C_N = [ A^{T} N^{-1} A ]^{-1}.
    """
    inv_mat_N = np.linalg.inv(mat_N)
    inv_map_covar = mat_A.T @ inv_mat_N @ mat_A
    map_covar = np.linalg.inv(inv_map_covar)
    mat_W = map_covar @ mat_A.T @ inv_mat_N
    if cov:
        return mat_W, map_covar
    return mat_W
