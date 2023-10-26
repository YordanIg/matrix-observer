"""
The entire forward-modelling formalism brought together.
"""
import numpy as np

import src.coordinates as CO
import src.spherical_harmonics as SH
import src.beam_functions as BF

def calc_averaging_matrix(Ntau, Nt):
    """
    Crude gain matrix to average up Nt neighbouring time bins into
    courser Ntau time bins. i.e. Ntau < Nt

    Exploits this trick for dividing each row by its own normalisation
    https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    """
    summation_matrix = calculate_summation_matrix(Ntau, Nt)

    #normalise each row to get averaging matrix
    averaging_matrix = summation_matrix / np.sum(summation_matrix, axis=1)[:, None]

    return averaging_matrix

def calc_zenith_driftscan_A(nside, lmax, Ntau, lat=-26, lon=0, Nt=np.linspace(0, 24, 24, endpoint=False)):
    """
    Calculate the total observation and binning matrix A = GPYB
    for a single drifscan antenna pointing at zenith with a cos^2
    beamfunction.
    """
    coords = CO.obs_zenith_drift_scan(lat, lon, Nt)
    mat_G = calc_averaging_matrix(Ntau=Ntau, lmax=lmax)
    mat_P = CO.calc_pointing_matrix(coords, nside=nside, pixels=False)
    mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)
    mat_B = BF.calc_beam_matrix(nside, lmax, beam_use=beam_cos)
    return mat_G @ mat_P @ mat_Y @ mat_B
    