"""
Reversing the forward-modelling process using maximum-likelihood methods.
"""
import numpy as np
from src.blockmat import BlockMatrix

def calc_ml_estimator_matrix(mat_A, mat_N, cov=False, delta=None, cond=False):
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

    If delta is passed, will L2 regularize by calculating
        W = [ A^{T} N^{-1} A + delta^2]^{-1} A^{T} N^{-1}
    instead.
    
    If cond=True, will calculate and print the condition number of the matrix
    A^{T} N^{-1} A
    """
    block_mats = False
    if isinstance(mat_A, BlockMatrix) and isinstance(mat_N, BlockMatrix):
        block_mats = True
    elif isinstance(mat_A, BlockMatrix) or isinstance(mat_N, BlockMatrix):
        raise TypeError(f"mat_A and mat_B must either both be ndarrays or BlockMatrix, but are {type(mat_A)} and {type(mat_N)}")
    
    if block_mats:
        inv_mat_N = mat_N.inv
    else:
        inv_mat_N = np.linalg.inv(mat_N)
    
    inv_map_covar = mat_A.T @ inv_mat_N @ mat_A

    # L2 regularization.
    if delta is not None:
        inv_map_covar += delta**2 *np.identity(len(inv_map_covar))
    
    if cond:
        if block_mats:
            print("1/condition #:", 1/np.linalg.cond(inv_map_covar.matrix))
        else:
            print("1/condition #:", 1/np.linalg.cond(inv_map_covar))
    
    if block_mats:
        map_covar = inv_map_covar.inv
    else:
        map_covar = np.linalg.inv(inv_map_covar)

    mat_W = map_covar @ mat_A.T @ inv_mat_N

    if cov:
        return mat_W, map_covar
    return mat_W
