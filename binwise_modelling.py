"""
Doing binwise multifrequency modelling to see if any of the methods we're
developing actually perform better.
"""
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from scipy.optimize import curve_fit

import src.beam_functions as BF
import src.spherical_harmonics as SH
import src.forward_model as FM
import src.sky_models as SM
from src.blockmat import BlockMatrix

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
    plt.plot(d.vector-mod(res[0]), '.')
    plt.ylabel("Temperature [K]")
    plt.show()


def fg_only_chrom():
    # Model and observation params
    nside   = 32
    lmax    = 32
    lats = np.array([-26])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 12, endpoint=False)
    Ntau  = 1
    nuarr   = np.linspace(50,100,51)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)

    # Generate observation matrix for the modelling and for the observations.
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau, lats, times, beam_use=BF.beam_cos_FWHM, chromaticity=BF.fwhm_func_tauscher)

    # Perform fiducial observations
    d = mat_A @ fg_alm
    #dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=1e7, seed=456)

    # Set up the foreground model
    Npoly = 5
    mod = FM.generate_binwise_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    res = curve_fit(mod_cf, nuarr, d.vector, p0=p0)
    print("std dev [mK]:", np.std(1e3*(d.vector-mod(res[0]))))
    plt.plot(d.vector, '.')
    plt.plot(mod(res[0]), '.')
    plt.ylabel("Temperature [K]")
    plt.show()
    plt.plot(d.vector-mod(res[0]), '.')
    plt.ylabel("Temperature [K]")
    plt.show()


def fg_cm21():
    # Model and observation params
    nside   = 32
    lmax    = 32
    Nlmax   = RS.get_size(lmax)
    lats = np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 12, endpoint=False)
    Nbin  = len(lats)*len(times)
    nuarr   = np.linspace(50,100,51)
    cm21_params     = [-0.2, 80.0, 5.0]
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
    dnoisy, noise_covar = SM.add_noise(d, dnu=1, Ntau=1, t_int=100, seed=456)
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")


    # Set up the foreground model
    Npoly = 3
    mod = FM.generate_binwise_cm21_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    p0 += cm21_params
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0, sigma=np.sqrt(noise_covar.diag))

    print("par est:", res[0])
    print("std devs:", np.sqrt(np.diag(res[1])))
    print("chi-sq")
    plt.plot(dnoisy.vector-mod(res[0]), '.')
    plt.xlabel("bin")
    plt.ylabel("data - model [K]")
    plt.show()

    # Provide a corner plot for the 21-cm inference.
    # Draw samples from the likelihood.
    chain = np.random.multivariate_normal(mean=res[0][-3:], cov=res[1][-3:,-3:], size=100000)
    c = ChainConsumer()
    c.add_chain(chain, parameters=['A', 'nu0', 'dnu'])
    f = c.plotter.plot()
    plt.show()

    # Evaluate the model at 100 points drawn from the chain to get 1sigma 
    # inference bounds in data space.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    chain_samples = np.random.multivariate_normal(mean=res[0][-3:], cov=res[1][-3:,-3:], size=100)
    cm21_a00_sample_list = [cm21_a00_mod(nuarr, theta) for theta in chain_samples]
    cm21_a00_sample_mean = np.mean(cm21_a00_sample_list, axis=0)
    cm21_a00_sample_std = np.std(cm21_a00_sample_list, axis=0)

    # Plot the model evaluated 1 sigma regions and the fiducial monopole.
    cm21_a00 = np.array(cm21_alm[::Nlmax])
    plt.plot(nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-cm21_a00_sample_std, 
        cm21_a00_sample_mean+cm21_a00_sample_std,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-2*cm21_a00_sample_std, 
        cm21_a00_sample_mean+2*cm21_a00_sample_std,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("21-cm a00 [K]")
    plt.legend()
    plt.show()

    # Do the same thing but take the residuals.
    plt.plot(nuarr, cm21_a00-cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-cm21_a00_sample_std-cm21_a00, 
        cm21_a00_sample_mean+cm21_a00_sample_std-cm21_a00,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-2*cm21_a00_sample_std-cm21_a00, 
        cm21_a00_sample_mean+2*cm21_a00_sample_std-cm21_a00,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Inferred 21-cm a00 residuals [K]")
    plt.legend()
    plt.show()

def fg_cm21_chrom():
    # Model and observation params
    nside   = 32
    lmax    = 32
    Nlmax   = RS.get_size(lmax)
    lats = [-26]#np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 6, 3)
    Ntau  = 1
    nuarr = np.linspace(50,100,51)
    cm21_params     = [-0.2, 80.0, 5.0]
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground and 21-cm alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm
    
    # Generate observation matrix for the modelling and for the observations.
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau, lats, times, beam_use=BF.beam_cos_FWHM, chromaticity=BF.fwhm_func_tauscher)
    
    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, dnu=1, Ntau=Ntau, t_int=100, seed=456)
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")


    # Set up the foreground model
    Npoly = 5
    mod = FM.generate_binwise_cm21_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    p0 += cm21_params
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0, sigma=np.sqrt(noise_covar.diag))

    print("par est:", res[0])
    print("std devs:", np.sqrt(np.diag(res[1])))
    print("chi-sq")
    plt.plot(dnoisy.vector-mod(res[0]), '.')
    plt.xlabel("bin")
    plt.ylabel("data - model [K]")
    plt.show()

    # Provide a corner plot for the 21-cm inference.
    # Draw samples from the likelihood.
    chain = np.random.multivariate_normal(mean=res[0][-3:], cov=res[1][-3:,-3:], size=100000)
    c = ChainConsumer()
    c.add_chain(chain, parameters=['A', 'nu0', 'dnu'])
    f = c.plotter.plot()
    plt.show()

    # Evaluate the model at 100 points drawn from the chain to get 1sigma 
    # inference bounds in data space.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    chain_samples = np.random.multivariate_normal(mean=res[0][-3:], cov=res[1][-3:,-3:], size=100)
    cm21_a00_sample_list = [cm21_a00_mod(nuarr, theta) for theta in chain_samples]
    cm21_a00_sample_mean = np.mean(cm21_a00_sample_list, axis=0)
    cm21_a00_sample_std = np.std(cm21_a00_sample_list, axis=0)

    # Plot the model evaluated 1 sigma regions and the fiducial monopole.
    cm21_a00 = np.array(cm21_alm[::Nlmax])
    plt.plot(nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-cm21_a00_sample_std, 
        cm21_a00_sample_mean+cm21_a00_sample_std,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-2*cm21_a00_sample_std, 
        cm21_a00_sample_mean+2*cm21_a00_sample_std,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("21-cm a00 [K]")
    plt.legend()
    plt.show()

    # Do the same thing but take the residuals.
    plt.plot(nuarr, cm21_a00-cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-cm21_a00_sample_std-cm21_a00, 
        cm21_a00_sample_mean+cm21_a00_sample_std-cm21_a00,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        nuarr,
        cm21_a00_sample_mean-2*cm21_a00_sample_std-cm21_a00, 
        cm21_a00_sample_mean+2*cm21_a00_sample_std-cm21_a00,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Inferred 21-cm a00 residuals [K]")
    plt.legend()
    plt.show()




