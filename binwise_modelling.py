"""
Doing binwise multifrequency modelling to see if any of the methods we're
developing actually perform better.
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

def fg_only():
    # Model and observation params
    nside   = 32
    lmax    = 32
    lats = np.array([-26])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 12, endpoint=False)
    Nbin  = len(lats)*len(times)
    nuarr   = np.linspace(50,100,51)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)

    # Generate observation matrix for the modelling and for the observations.
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=1, lats=lats, times=times, beam_use=narrow_cosbeam)
    mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
    
    # Perform fiducial observations
    d = mat_A @ fg_alm
    #dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=1e7, seed=456)

    # Set up the foreground model
    mod = FM.generate_binwise_forward_model(nuarr, mat_A, Npoly=3)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    res = curve_fit(mod_cf, nuarr, d.vector, p0=[10, -2.5, 0.01])
    print("std dev [mK]:", np.std(1e3*(d.vector-mod(res[0]))))
    plt.plot(d.vector, '.')
    plt.plot(mod(res[0]), '.')
    plt.ylabel("Temperature [K]")
    plt.show()
    plt.plot(1e3*(d.vector-mod(res[0])), '.')
    plt.ylabel("Temperature [mK]")
    plt.show()


def fg_cm21():
    # Model and observation params
    nside   = 32
    lmax    = 32
    delta   = 1e-3
    Nlmax   = RS.get_size(lmax)
    npix    = hp.nside2npix(nside)
    lats = np.array([-26])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 12, endpoint=False)
    Nbin  = len(lats)*len(times)
    nuarr   = np.linspace(50,100,51)
    cm21_params     = (-0.2, 80.0, 5.0)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground and 21-cm alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm
    
    # Generate observation matrix for the modelling and for the observations.
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=1, lats=lats, times=times, beam_use=narrow_cosbeam)
    mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
    
    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=1e4, seed=456)

    # Set up the foreground model
    mod = FM.generate_binwise_cm21_forward_model(nuarr, mat_A, Npoly=3)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=[10, -2.5, 0.01, -.2, 80.0, 5.0])

    print("par est:", res[0])
    print("std devs:", np.sqrt(np.diag(res[1])))
    print("chi-sq")

    plt.plot(dnoisy.vector, '.')
    plt.plot(mod(res[0]), '.')
    plt.show()
    plt.plot(dnoisy.vector-mod(res[0]), '.')
    plt.show()

    cm21_a00 = np.array(cm21_alm[::Nlmax])
    plt.plot(np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *res[0][-3:]), label="reconstructed 21-cm a00")
    plt.plot(cm21_a00, label='fiducial 21-cm a00', linestyle=':', color='k')
    plt.legend()
    plt.show()
    plt.plot(np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *res[0][-3:])-cm21_a00, label="reconstructed 21-cm a00 residuals")
    plt.axhline(y=0, linestyle=':', color='k')
    plt.legend()
    plt.show()




