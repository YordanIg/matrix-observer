"""
Using maximum-likelihood methods to reconstruct a_{00}(\nu), then fit a power
law and a 21-cm signal to it.
"""
from pygdsm import GlobalSkyModel2016
import healpy as hp
import numpy as np
from numpy.linalg import svd
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

import src.beam_functions as BF
import src.spherical_harmonics as SH
import src.forward_model as FM
import src.sky_models as SM
import src.map_making as MM
import src.plotting as PL
from src.blockmat import BlockMatrix, BlockVector

RS = SH.RealSphericalHarmonics()

# Fit the foreground and 21-cm monopole.
def fg_polymod(nuarr, *theta_fg):
    Afg, alpha = theta_fg[:2]
    zetas      = theta_fg[2:]
    exponent = [zetas[i]*np.log(nuarr/60)**(i+2) for i in range(len(zetas))]
    fg_a00_terms = (Afg*1e3)*(nuarr/60)**(-alpha) * np.exp(np.sum(exponent, 0))
    return fg_a00_terms

def cm21_mod(nuarr, *theta_21):
    A21, nu0, dnu = theta_21
    cm21_a00_terms = np.sqrt(4*np.pi) * A21 * np.exp(-.5*((nuarr-nu0)/dnu)**2)
    return cm21_a00_terms

def fg_cm21_polymod(nuarr, *theta):
    theta_fg = theta[:-3]
    theta_21 = theta[-3:]
    return fg_polymod(nuarr, *theta_fg) + cm21_mod(nuarr, *theta_21)

################################################################################
def trivial_obs():
    # Model and observation params
    nside   = 16
    lmax    = 32
    lmod    = lmax
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)
    npix    = hp.nside2npix(nside)
    nuarr   = np.linspace(50,100,51)
    cm21_params     = (-0.2, 80.0, 5.0)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground and 21-cm signal alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix
    mat_A = FM.calc_observation_matrix_all_pix(nside, lmax, Ntau=npix, beam_use=narrow_cosbeam)
    mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))

    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=npix, t_int=1e4)

    # Reconstruct the max likelihood estimate of the alm
    mat_W   = MM.calc_ml_estimator_matrix(mat_A, noise_covar)
    rec_alm = mat_W @ dnoisy

    # Extract the monopole component of the reconstructed alm.
    fg_a00  = np.array(fg_alm[::Nlmax])
    rec_a00 = np.array(rec_alm.vector[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    fg_mon_p0 = [15, 2.5, .001]
    cm21_mon_p0 = [-0.2, 80, 5]
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, p0=fg_mon_p0+cm21_mon_p0)
    
    # Plot everything
    plt.plot(nuarr, cm21_mod(nuarr, *res[0][-3:]), label='fit 21-cm monopole')
    plt.plot(nuarr, rec_a00-fg_a00, label='$a_{00}$ reconstructed - fid fg')
    plt.plot(nuarr, cm21_mod(nuarr, *cm21_mon_p0), label='fiducial 21-cm monopole', linestyle=':', color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()

    plt.plot(nuarr, cm21_mod(nuarr, *res[0][-3:])-cm21_mod(nuarr, *cm21_mon_p0), label='fit 21-cm monopole')
    plt.plot(nuarr, rec_a00-fg_a00-cm21_mod(nuarr, *cm21_mon_p0), label='$a_{00}$ reconstructed - fid fg')
    plt.plot(nuarr, cm21_mod(nuarr, *cm21_mon_p0)-cm21_mod(nuarr, *cm21_mon_p0), label='fiducial 21-cm monopole', linestyle=':', color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()


def nontrivial_obs():
    # Model and observation params
    nside   = 32
    lmax    = 32
    lmod    = 4
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)
    npix    = hp.nside2npix(nside)
    lats = np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)
    times = np.linspace(0, 24, 24, endpoint=False)
    nuarr   = np.linspace(50,100,51)
    cm21_params     = (-0.2, 80.0, 5.0)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground and 21-cm signal alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix for the modelling and for the observations.
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, lats=lats, times=times, beam_use=narrow_cosbeam)
    mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
    mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmod, lats=lats, times=times, beam_use=narrow_cosbeam)
    mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))
    

    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=npix, t_int=1e4)

    # Reconstruct the max likelihood estimate of the alm
    mat_W   = MM.calc_ml_estimator_matrix(mat_A_mod, noise_covar)
    rec_alm = mat_W @ dnoisy

    # Extract the monopole component of the reconstructed alm.
    fg_a00  = np.array(fg_alm[::Nlmax])
    rec_a00 = np.array(rec_alm.vector[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    fg_mon_p0 = [15, 2.5, .001]
    cm21_mon_p0 = [-0.2, 80, 5]
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, p0=fg_mon_p0+cm21_mon_p0)
    
    # Plot everything
    plt.plot(nuarr, cm21_mod(nuarr, *res[0][-3:]), label='fit 21-cm monopole')
    #plt.plot(nuarr, rec_a00-fg_a00, label='$a_{00}$ reconstructed - fid fg')
    plt.plot(nuarr, cm21_mod(nuarr, *cm21_mon_p0), label='fiducial 21-cm monopole', linestyle=':', color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()

    plt.plot(nuarr, cm21_mod(nuarr, *res[0][-3:])-cm21_mod(nuarr, *cm21_mon_p0), label='fit 21-cm monopole')
    #plt.plot(nuarr, rec_a00-fg_a00-cm21_mod(nuarr, *cm21_mon_p0), label='$a_{00}$ reconstructed - fid fg')
    plt.plot(nuarr, cm21_mod(nuarr, *cm21_mon_p0)-cm21_mod(nuarr, *cm21_mon_p0), label='fiducial 21-cm monopole', linestyle=':', color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()
