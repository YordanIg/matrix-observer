"""
Reversing the forward-modelling process using maximum-likelihood methods.
"""
import numpy as np
from scipy.special import eval_legendre
import healpy as hp
from src.blockmat import BlockMatrix

def calc_reg_matrix_exp(Nlmod, nuarr, dnu=1, pow=2.5):
    """
    Calculate the regularization matrix B from that textbook, but made for 
    exponentials.
    """
    # Make a single block of the matrix.
    gammas = [1-dnu*pow/nu for nu in nuarr]
    gammas.pop()
    ones = [1.]*(len(nuarr)-2)
    final_column = np.zeros(len(nuarr)-1)
    final_column[-1] = 1.
    square_block = -np.diag(gammas) + np.diag(ones, k=1)
    block = np.c_[square_block, final_column]
    
    # Construct the block matrix.
    return BlockMatrix(mat=block, mode='block', nblock=Nlmod).matrix


def sort_alm_vector_frequency(a, Nfreq, Nlmod):
    """
    Sorts an alm vector from an ordering of 
        (a00(freq1), a1-1(freq1), ..., a00(freq2), ...)
    to an ordering
        (a00(freq1), a00(freq2), ..., a1-1(freq1), ...)
    i.e. from blocks of different l and m modes to blocks of different 
    frequencies.
    """
    assert len(a) == Nfreq*Nlmod
    sorted_a = [a[i::Nlmod] for i in range(Nlmod)]
    return np.array(sorted_a).flatten()


def sort_alm_vector_lm(a, Nfreq, Nlmod):
    """
    Sorts an alm vector from an ordering of 
        (a00(freq1), a00(freq2), ..., a1-1(freq1), ...)
    to an ordering
        (a00(freq1), a1-1(freq1), ..., a00(freq2), ...)
    i.e. from blocks of different frequencies to blocks of different l and m 
    modes.
    """
    assert len(a) == Nfreq*Nlmod
    sorted_a = []
    for i in range(Nfreq):
        sorted_a += list(a[i::Nfreq])
    return np.array(sorted_a)


def calc_ml_estimator_matrix(mat_A, mat_N, cov=False, delta=None, reg='L2', nuarr=None, cond=False, pow=None):
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
        Nlmod = mat_A.block_shape[1]
    elif isinstance(mat_A, BlockMatrix) or isinstance(mat_N, BlockMatrix):
        raise TypeError(f"mat_A and mat_B must either both be ndarrays or BlockMatrix, but are {type(mat_A)} and {type(mat_N)}")
    
    if block_mats:
        inv_mat_N = mat_N.inv
    else:
        inv_mat_N = np.linalg.inv(mat_N)
    
    inv_map_covar = mat_A.T @ inv_mat_N @ mat_A

    # Regularization.
    if delta is not None:
        if reg=='L2':
            cost_mat_H = np.identity(len(inv_map_covar))
        elif reg=='exp':
            if block_mats:
                print("Exponential regularization - switching from BlockMats to ndarrays.")
                block_mats    = False
                inv_map_covar = inv_map_covar.matrix
                mat_A = mat_A.matrix
                inv_mat_N = inv_mat_N.matrix

            # Calculate the B cost matrix.
            cost_mat_B = calc_reg_matrix_exp(Nlmod=Nlmod, nuarr=nuarr, dnu=nuarr[1]-nuarr[0], pow=pow)
            # Permute it to take it from alm blocks to frequency blocks.
            cost_mat_B = sort_alm_vector_lm(a=cost_mat_B.T, Nfreq=len(nuarr), Nlmod=Nlmod).T
            cost_mat_H = cost_mat_B.T @ cost_mat_B
        else:
            raise ValueError
        inv_map_covar += delta**2 * cost_mat_H
    
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


def calc_full_unmodelled_mode_matrix(lmod, lmax, nside, foreground_power_spec, 
                                     beam_mat, binning_mat, pointing_mat):
    """
    Calculate the unmodelled mode matrix for the non-trivial observation 
    strategy and binning of timeseries data into Ntau bins.
    """
    npix = hp.nside2npix(nside)
    vectors = hp.pix2vec(nside, ipix=list(range(npix)))
    vectors = np.array(vectors).T
    vector_difference = np.einsum("pi,qi->pq", vectors, vectors)
    binpoint_mat = binning_mat@pointing_mat
    val = np.sum([((2*l+1)/(4*np.pi)) * eval_legendre(l, vector_difference) * foreground_power_spec[l] * beam_mat[l,l]**2 for l in range(lmod+1, lmax)], axis=0)
    return binpoint_mat@val@(binpoint_mat.T)


def calc_full_unmodelled_mode_matrix_multifreq(lmod, lmax, nside, 
                                               foreground_power_spec_list, 
                                     beam_mat, binning_mat, pointing_mat):
    """
    Calculate the unmodelled mode matrix for the non-trivial observation 
    strategy and binning of timeseries data into Ntau bins for Nfreq block
    matrices.
    """
    npix = hp.nside2npix(nside)
    vectors = hp.pix2vec(nside, ipix=list(range(npix)))
    vectors = np.array(vectors).T
    vector_difference = np.einsum("pi,qi->pq", vectors, vectors)
    binpoint_mat = binning_mat.block[0]@pointing_mat.block[0]
    legendre_list_zeros = np.zeros(shape=(lmod+1, npix, npix))
    legendre_list_values = [eval_legendre(l, vector_difference) for l in range(lmod+1, lmax)]
    legendre_list = np.append(legendre_list_zeros, legendre_list_values, axis=0)

    mat_S = []
    for mat_B_block, fps in zip(beam_mat.block, foreground_power_spec_list):
        print("bam")
        val = np.sum([((2*l+1)/(4*np.pi)) * legendre_list[l] * fps[l] * mat_B_block[l,l]**2 for l in range(lmod+1, lmax)], axis=0)
        mat_S.append(binpoint_mat@val@(binpoint_mat.T))

    return BlockMatrix(mat=np.array(mat_S))
