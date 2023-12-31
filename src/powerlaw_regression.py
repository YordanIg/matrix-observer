"""
Fitting models onto reconstructed alm to separate the foreground alm and signal
alm.
"""
import numpy as np
from numpy.linalg import inv
from scipy.linalg import svd
from scipy.optimize import minimize
from emcee import EnsembleSampler
from src.sky_models import cm21_globalT


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

    Returns : X' (N, Nmod+2)
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


'''def powerlaw_forward_model(theta, x):
    """
    Forward model the power law:
        Ax^(-gamma)
    for theta = (A, gamma).
    """
    amplitude, gamma = theta
    return amplitude * x **(gamma)

def powerlaw_regression(y, x, theta0):
    """
    Find theta = (amplitude, power law index) that zeros:
        powerlaw_forward_model(theta, x) - y
    given a starting guess theta0.
    """
    to_min = '''




def fg_powerlawPCA_forward_model(nuarr, theta_fg, pca_basis):
    """
    Forward model the power law + PCA foreground model, given an (Nfreq, Npca) 
    basis matrix pca_basis. The vector theta_fg consists of:
        amplitude, power law index, pca params
    """
    assert len(nuarr) == np.shape(pca_basis)[0]
    assert len(theta_fg) == np.shape(pca_basis)[1]+2
    amplitude, slope = theta_fg[:2]
    theta_pca = theta_fg[2:]
    return amplitude * (nuarr/60) **(slope) * np.exp(slope* pca_basis@theta_pca )


def fg_powerlawPCA_regression(nuarr, C0, pca_basis, bounds):
    """
    Fit a power law + PCA to data. 
    """
    Npca = np.shape(pca_basis)[1]
    to_max = lambda theta: -np.sum((C0 - fg_powerlawPCA_forward_model(nuarr, theta, pca_basis))**2)

    x0 = [np.exp(40), -5]
    x0 += [1e-4]*Npca

    nwalkers = 16
    ndim = 2 + Npca

    pos = np.array([np.random.uniform(*bound, size=nwalkers) for bound in bounds])
    pos = pos.T

    print(np.shape(pos))
    sampler = EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=to_max)
    sampler.run_mcmc(pos, 10000, progress=True)
    return sampler


def fg_powerlawPCA_cm21mon_forward_model(nuarr, theta, pca_basis):
    """
    Forward model the power law + PCA foreground model and the 21-cm gaussian, 
    given an (Nfreq, Npca) basis matrix pca_basis. The vector theta_fg consists 
    of:
        amplitude, power law index, pca params, 21-cm params
    """
    assert len(theta) == np.shape(pca_basis)[1] + 2 + 3
    amplitude, slope = theta[:2]
    theta_pca = theta[2:-3]
    gauss_amp = theta[-3]
    gauss_cent = theta[-2]
    gauss_width = theta[-1]
    fg_part = amplitude * (nuarr/60) **(slope) * np.exp(slope* pca_basis@theta_pca )
    cm21_part = cm21_globalT(nuarr, gauss_amp, gauss_cent, gauss_width)
    return fg_part+cm21_part


def fg_powerlawPCA_cm21mon_regression(nuarr, C0, pca_basis, bounds):
    """
    Fit a power law + PCA and a gaussian trough to data. Returns an emcee 
    sampler that's been run.
    """
    Npca = np.shape(pca_basis)[1]
    to_max = lambda theta: -np.sum((C0 - fg_powerlawPCA_cm21mon_forward_model(nuarr, theta, pca_basis))**2)

    x0 = [np.exp(40), -5]
    x0 += [1e-4]*Npca

    nwalkers = 16
    ndim = 2 + Npca + 3

    pos = np.array([np.random.uniform(*bound, size=nwalkers) for bound in bounds])
    pos = pos.T

    print(np.shape(pos))
    sampler = EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=to_max)
    sampler.run_mcmc(pos, 100000, progress=True)
    return sampler

