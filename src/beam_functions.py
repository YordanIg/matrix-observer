"""
Functions for calculating the beam matrix for an azimuthally symmetric beam.

Goal is to allow easy convolution via:

    convolved spatial map, d = Y.B.a

Where B is the beam matrix calculated here.
      Y is the spherical harmonic matrix
      a is the vector of real valued alm
"""
import healpy
import numpy as np
import numpy.random as npr
import scipy as sp
import h5py
import astropy
import matplotlib.pyplot as plt

import src.spherical_harmonics as SH

def beam_cos(theta, theta0=1.0):
    """
    Example beam function. Azimuthally symmetrix beam
    as can not depend on phi.

    Define a single lobe cos^2 beam.

    theta0 allows narrowing of the beam
    """
    if theta < np.pi / 2.0 * theta0:
        b = np.cos(theta / theta0)**2
        b *= 3.0/2.0/np.pi
    else:
        b = 0.0
    return b

def beam_gauss(theta, theta0=0.1):
    """
    Example beam function. Azimuthally symmetrix beam
    as can not depend on phi.

    Define a single lobe cos^2 beam
    """
    if theta < np.pi/2.0:
        b = np.exp(-(theta/theta0)**2/2.0)
    else:
        b = 0.0
    return b

def beam_map(nside, beam=beam_cos, norm_flag=True):
    """
    Calculate the beam map for an aximuthally symmetric beam

    Note that beam can only depend upon theta by design.
    """
    #create an empty map and then fill it in with the desired beam shape
    npix = healpy.nside2npix(nside)
    map_raw = np.zeros(npix)
    for i in range(npix):
        theta, phi = healpy.pixelfunc.pix2ang(nside, i)
        map_raw[i] = beam(theta)

    #enforce beam normalisation so integrates over full sky to one
    if norm_flag:
        norm = np.sum(map_raw)*healpy.nside2pixarea(nside)
        map_raw = map_raw / norm
    return map_raw

def calc_blm(nside, lmax, beam_use=beam_cos, norm_flag=True):
    """
    Get the real valued blm for the beam
    """
    #calculate the map and transform it to get the blm
    map_raw = beam_map(nside, beam_use, norm_flag)
    blm = healpy.sphtfunc.map2alm(map_raw, lmax=lmax, use_weights=False)

    #Make a clean beam map with no off diagonals to test reconstruction
    blm_clean = np.zeros(blm.shape,dtype=np.complex128)
    blm_clean[0:lmax+1] = blm[0:lmax+1]

    #convert complex blm into real blm
    RS = SH.RealSphericalHarmonics()
    blm_real = RS.Complex2RealALM(blm_clean)

    return blm_real

def calc_beam_matrix(nside, lmax, beam_use=beam_cos, norm_flag=True):
    """
    Get the beam matrix for an azimuthally symmetric beam

    For getting the convolution right, I seem to need a factor of
    np.sqrt(4.0*np.pi/(2.0*l+1))
    for the bl. I understand where this comes from, but not why
    it's not mentioned in some of the papers.

    returns:
        beam_matrix - (nalm x nalm) real matrix with bl on diagonal
    """
    RS = SH.RealSphericalHarmonics()

    #get the real blm of the map
    blm_real = calc_blm(nside, lmax, beam_use, norm_flag)

    #form the beam matrix from the blm
    nalm = RS.get_size(lmax)
    beam_matrix = np.zeros((nalm, nalm))
    for i in range(nalm):
        (l,m) = RS.get_lm(i)
        bindx = RS.get_idx(l,0)
        bl = blm_real[bindx] * np.sqrt(4.0*np.pi/(2.0*l+1))
        beam_matrix[i,i] = bl
    return beam_matrix

def _test_convolution(map_input = None, nside = 8, lmax = 20, beam=beam_cos):
    """
    Convolution of beam with a constant map should return the input map

    Not the most detailed test, but a useful one.
    """
    #make a constant map if none given
    if map_input is None:
        npix = healpy.nside2npix(nside)
        map_input = 100.0 * np.ones(npix)

    #calculate the real alm of the constant map
    RS = TA.RealSphericalHarmonics()
    alm = healpy.sphtfunc.map2alm(map_input, lmax=lmax, use_weights=False)
    alm_real = RS.Complex2RealALM(alm)

    #calculate beam matrix
    beam_matrix = calc_beam_matrix(nside, lmax, beam)

    #convolve by multiplication and map back to space
    Y = SH.calculate_Y(nside=nside, lmax=lmax)
    map_convolve = Y @ beam_matrix @ alm_real

    #compare input and recovered maps
    residuals = map_input - map_convolve

    print(residuals)

    return residuals, map_input, map_convolve, beam_matrix
