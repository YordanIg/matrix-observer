"""
The entire forward-modelling formalism brought together.
"""
import numpy as np

import src.coordinates as CO
import src.spherical_harmonics as SH
import src.beam_functions as BF
from src.blockmat import BlockMatrix

def _calc_summation_matrix(Ntau, Nt):
    """
    Crude gain matrix to sum up Nt neighbouring time bins into
    courser Ntau time bins. i.e. Ntau < Nt
    """
    summation_matrix = np.zeros([Ntau, Nt])
    for i in range(Nt):
        value = int(i * Ntau / Nt)
        summation_matrix[value][i] = 1

    return summation_matrix

def calc_averaging_matrix(Ntau, Nt):
    """
    Crude gain matrix to average up Nt neighbouring time bins into
    courser Ntau time bins. i.e. Ntau < Nt

    Exploits this trick for dividing each row by its own normalisation
    https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    """
    summation_matrix = _calc_summation_matrix(Ntau, Nt)

    #normalise each row to get averaging matrix
    averaging_matrix = summation_matrix / np.sum(summation_matrix, axis=1)[:, None]

    return averaging_matrix

################################################################################
# Single-frequency observation matrices.
################################################################################

def calc_observation_matrix_zenith_driftscan(nside, lmax, Ntau=None, lat=-26, lon=0, 
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a single drifscan antenna pointing at zenith with a cos^2
    beamfunction. If Ntau = len(times), G is just the identity matrix.

    If return_mat is True, function returns tuple of G,P,Y,B too.
    """
    if Ntau is None:  # If Ntau isn't passed, don't bin.
        Ntau = len(times)
    coords = CO.obs_zenith_drift_scan(lat, lon, times)
    mat_G = calc_averaging_matrix(Ntau=Ntau, Nt=len(times))
    mat_P = CO.calc_pointing_matrix(coords, nside=nside, pixels=False)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    if return_mat:
        return mat_G @ mat_P @ mat_Y @ mat_B, (mat_G, mat_P, mat_Y, mat_B)
    return mat_G @ mat_P @ mat_Y @ mat_B

def calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=None, lats=[-26],
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a single drifscan antenna pointing at zenith with a cos^2
    beamfunction. If Ntau = len(times), G is just the identity matrix.

    If return_mat is True, function returns tuple of G,P,Y,B too.
    """
    if Ntau is None:  # If Ntau isn't passed, don't bin.
        Ntau = len(times)*len(lats)
    coords = [CO.obs_zenith_drift_scan(lat, lon=0, times=times) for lat in lats]
    mat_G = calc_averaging_matrix(Ntau=Ntau, Nt=len(times)*len(lats))
    mat_P = CO.calc_pointing_matrix(*coords, nside=nside, pixels=False)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    if return_mat:
        return mat_G @ mat_P @ mat_Y @ mat_B, (mat_G, mat_P, mat_Y, mat_B)
    return mat_G @ mat_P @ mat_Y @ mat_B

def calc_observation_matrix_all_pix(nside, lmax, Ntau, beam_use=BF.beam_cos, 
                                    return_mat=False):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a hypothetical antenna experiment that can point at every pixel once.
    If Ntau = len(times), G is just the identity matrix.

    If spherical_harmonic_mat is True, function returns it too.
    """
    #pointing matrix is just the identity matrix, so not included
    npix = 12*nside**2
    mat_G = calc_averaging_matrix(Ntau=Ntau, Nt=npix)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    if return_mat:
        return mat_G @ mat_Y @ mat_B, (mat_G, mat_Y, mat_B)
    return mat_G @ mat_Y @ mat_B

################################################################################
# Multi-frequency observation matrices.
################################################################################

def calc_observation_matrix_zenith_driftscan_multifreq(nuarr, nside, lmax, Ntau=None, 
                            lat=-26, lon=0, 
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Do the same thing as calc_observation_matrix_zenith_driftscan but return 
    multifrequency block matrices.
    """
    mats = calc_observation_matrix_zenith_driftscan(nside, lmax, Ntau=Ntau, 
                            lat=lat, lon=lon, 
                            times=times, 
                            beam_use=beam_use, return_mat=return_mat)
    if return_mat:
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = mats
        mat_A_bl = BlockMatrix(mat=mat_A, nblock=len(nuarr))
        mat_G_bl = BlockMatrix(mat=mat_G, nblock=len(nuarr))
        mat_P_bl = BlockMatrix(mat=mat_P, nblock=len(nuarr))
        mat_Y_bl = BlockMatrix(mat=mat_Y, nblock=len(nuarr))
        mat_B_bl = BlockMatrix(mat=mat_B, nblock=len(nuarr))
        return mat_A_bl, (mat_G_bl, mat_P_bl, mat_Y_bl, mat_B_bl)
    return BlockMatrix(mat=mat_A, nblock=len(nuarr))

def calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr, nside, lmax, Ntau=None, lats=[-26],
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos, return_mat=False):
    """
    Do the same thing as calc_observation_matrix_multi_zenith_driftscan but 
    return multifrequency block matrices.
    """
    mats = calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=Ntau, lats=lats,
                            times=times, 
                            beam_use=beam_use, return_mat=return_mat)
    if return_mat:
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = mats
        mat_A_bl = BlockMatrix(mat=mat_A, nblock=len(nuarr))
        mat_G_bl = BlockMatrix(mat=mat_G, nblock=len(nuarr))
        mat_P_bl = BlockMatrix(mat=mat_P, nblock=len(nuarr))
        mat_Y_bl = BlockMatrix(mat=mat_Y, nblock=len(nuarr))
        mat_B_bl = BlockMatrix(mat=mat_B, nblock=len(nuarr))
        return mat_A_bl, (mat_G_bl, mat_P_bl, mat_Y_bl, mat_B_bl)
    return BlockMatrix(mat=mat_A, nblock=len(nuarr))

def calc_observation_matrix_all_pix_multifreq(nuarr, nside, lmax, Ntau, 
                                              beam_use=BF.beam_cos, 
                                              return_mat=False):
    """
    Do the same thing as calc_observation_matrix_all_pix but return 
    multifrequency block matrices.
    """
    mats = calc_observation_matrix_all_pix(nside, lmax, Ntau, 
                                           beam_use=beam_use, 
                                           return_mat=return_mat)
    if return_mat:
        mat_A, (mat_G, mat_Y, mat_B) = mats
        mat_A_bl = BlockMatrix(mat=mat_A, nblock=len(nuarr))
        mat_G_bl = BlockMatrix(mat=mat_G, nblock=len(nuarr))
        mat_Y_bl = BlockMatrix(mat=mat_Y, nblock=len(nuarr))
        mat_B_bl = BlockMatrix(mat=mat_B, nblock=len(nuarr))
        return mat_A_bl, (mat_G_bl, mat_Y_bl, mat_B_bl)
    return BlockMatrix(mat=mat_A, nblock=len(nuarr))
