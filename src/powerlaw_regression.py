"""
Fitting models onto reconstructed alm to separate the foreground alm and signal
alm.
"""
import numpy as np
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


def pca_regression(y, basis_mat, p_guess):
    """
    Given an (Npca, N) basis matrix, find the (Npca,) vector p which minimizes
    the cost function:
        a . basis_mat - y
    where d is a vector of shape (N,).
    
    p_guess is the initual guess for the vector p.
    """
    assert np.shape(basis_mat)[0] == len(p_guess)
    assert np.shape(basis_mat)[1] == len(y)

    to_min = lambda a: np.sum((a @ basis_mat - y)**2)
    res = minimize(to_min, x0=p_guess, method="Nelder-Mead", tol=1e-10)
    if not res.success:
        raise Exception("did not converge.")
    return res.x

