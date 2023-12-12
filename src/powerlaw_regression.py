"""
Fitting models onto reconstructed alm to separate the foreground alm and signal
alm.
"""
import numpy as np
from numpy.linalg import inv
from scipy.linalg import svd
from scipy.optimize import minimize


def pca(arr, N):
    """
    Represent the rows of an (M, N) array in an Npca-dimensional 
    coordinate system, where Npca < N.

    Returns
    -------
    coordinate matrix
        The rows of the matrix correspond to the coordinates
        of each row of arr in the reduced dimension space.
    basis matrix
        (Npca, N) matrix with columns representing the basis vectors of
        the decomposition.
    """
    # Carry out SVD.
    U, S, Vh = svd(arr, full_matrices=False)

    # Truncate the SVD matrices.
    u_matrix = U[:,:N]
    singular_matrix = np.diag(S[:N])

    # Compute the coordinate and basis matrices.
    coordinate_matrix = u_matrix @ singular_matrix
    basis_matrix = Vh[:N]

    return coordinate_matrix, basis_matrix


def power_law_residuals(x, y):
    """
    Calculate the residuals of the power law collapse of log(y), returning these
    as well as the array log(x) and the power law amplitude and slope.

    Assuming that:
        log(y) = amplitude + slope*log(x) + slope*residuals
    this function returns the tuple:
        (log(x), residuals, fit_params)
    for fit_params containing amplitude and slope.

    Parameters
    ----------
    x : (N,) array
    y : (N,) or (M, N) array

    Returns
    -------
    log(x) : (N,) array
    residuals : (N,) or (M, N) array
        The residuals of the power-law fit.
    fit_params : (2,) or (M, 2) array
        The (amplitude, slope) of the straight-line fit in log-log space.
    """
    assert len(np.shape(y)) in [1, 2]
    
    x_log = np.log(x)
    y_log = np.log(y).T

    # Fit straight lines to the power laws and collapse onto y=x line.
    slopes, amplitudes = np.polyfit(x_log, y_log, deg=1)
    y_log_collapsed = y_log-amplitudes
    y_log_collapsed /= slopes

    # Minus the frequency array to collapse onto y=0.
    residuals = y_log_collapsed.T-x_log

    fit_info = np.array([amplitudes, slopes]).T
    return x_log, residuals, fit_info


def noisefree_linear_regression(y, mat_X):
    """
    Given an (N, Nmod) matrix X, find the (Nmod,) vector theta which 
    minimizes the cost function:
        (X . theta) - y
    where y is a vector of shape (N,).
    """
    mat_W = inv( mat_X.T @ mat_X ) @ mat_X.T
    return mat_W @ y


def lin_pca_forward_mod(x, mat_X):
    """
    Given an (N, Nmod) matrix X, (Nmod,) and the parameters m, c, build a matrix
    X' that linearises the model:
        y = m x  +  c  +  m X . theta
    into the form
        y = X' . theta
    where y is a vector of shape (N,).

    Returns : X (N, Nmod+2)
    """
    N = len(x)
    Nmod = np.shape(mat_X)[1]
    assert np.shape(mat_X)[0] == N

    # Linearize the problem into the form y = X' . theta
    mat_X_new = np.zeros((N, Nmod+2))
    mat_X_new[:,0] = x
    mat_X_new[:,1] = [1]*N
    mat_X_new[:,2:] = mat_X

    return mat_X_new


def lin_pca_regression(x, y, mat_X):
    """
    Given an (N, Nmod) matrix X, and a (N,) vector x, find the (Nmod,) vector 
    theta and the parameters m, c which minimize the cost function:
        (m x  +  c  +  m X . theta)  -  y
    where y is a vector of shape (N,).

    Returns : (m, c, theta_1, theta_2, ...)
    """
    N = len(x)
    assert np.shape(mat_X)[0] == N
    assert len(y) == N

    # Linearize the problem into the form y = X' . theta
    mat_X_new = lin_pca_forward_mod(x, mat_X)
    return noisefree_linear_regression(y=y, mat_X=mat_X_new)


