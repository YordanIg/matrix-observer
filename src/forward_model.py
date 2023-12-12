"""
The entire forward-modelling formalism brought together.
"""
import numpy as np

import src.coordinates as CO
import src.spherical_harmonics as SH
import src.beam_functions as BF

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

def calc_observation_matrix_zenith_driftscan(nside, lmax, Ntau, lat=-26, lon=0, 
                            times=np.linspace(0, 24, 24, endpoint=False), 
                            beam_use=BF.beam_cos):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a single drifscan antenna pointing at zenith with a cos^2
    beamfunction. If Ntau = len(times), G is just the identity matrix.
    """
    coords = CO.obs_zenith_drift_scan(lat, lon, times)
    mat_G = calc_averaging_matrix(Ntau=Ntau, lmax=lmax)
    mat_P = CO.calc_pointing_matrix(coords, nside=nside, pixels=False)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    return mat_G @ mat_P @ mat_Y @ mat_B

def calc_observation_matrix_all_pix(nside, lmax, Ntau, Nt, beam_use=BF.beam_cos):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a hypothetical antenna experiment that can point at every pixel once.
    If Ntau = len(times), G is just the identity matrix.
    """
    #pointing matrix is just the identity matrix, so not included
    mat_G = calc_averaging_matrix(Ntau=Ntau, Nt=Nt)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_use)
    return mat_G @ mat_Y @ mat_B
