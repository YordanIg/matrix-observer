"""
Doing binwise multifrequency modelling to see if any of the methods we're
developing actually perform better.
"""
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from scipy.optimize import curve_fit
from emcee import EnsembleSampler
from chainconsumer import ChainConsumer
#import pypolychord

import src.beam_functions as BF
import src.spherical_harmonics as SH
import src.forward_model as FM
import src.sky_models as SM
import nregions_inference as NRI
import src.inference as INF
import src.observing as OBS
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
    plt.plot(d.vector-mod(res[0]), '.')
    plt.ylabel("Temperature [K]")
    plt.show()


def fg_only_chrom(mcmc=False):
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
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=1e7, seed=456)

    # Set up the foreground model
    Npoly = 5
    mod = FM.generate_binwise_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0)
    if mcmc:
        # create a small ball around the MLE the initialize each walker
        nwalkers, fg_dim = 64, Npoly
        ndim = fg_dim
        pos = res[0]*(1 + 1e-4*np.random.randn(nwalkers, ndim))

        # run emcee without priors
        err = np.sqrt(noise_covar.diag)
        sampler = EnsembleSampler(nwalkers, ndim, NRI.log_likelihood, 
                            args=(dnoisy.vector, err, mod))
        _=sampler.run_mcmc(pos, nsteps=3000, progress=True)
        chain = sampler.get_chain(flat=True, discard=1000)
        theta_inferred = np.mean(chain, axis=0)
        c = ChainConsumer()
        c.add_chain(chain)
        f = c.plotter.plot()
        plt.show()

    print("std dev [mK]:", np.std(1e3*(dnoisy.vector-mod(res[0]))))
    plt.plot(dnoisy.vector, '.')
    plt.plot(mod(res[0]), '.')
    plt.ylabel("Temperature [K]")
    plt.show()
    plt.plot(dnoisy.vector-mod(res[0]), '.')
    if mcmc:
        plt.plot(dnoisy.vector-mod(theta_inferred), '.')
    plt.ylabel("Temperature [K]")
    plt.show()


def fg_cm21(Npoly=3, mcmc=False):
    # Model and observation params
    nside   = 32
    lmax    = 32
    Nlmax   = RS.get_size(lmax)
    lats = np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = times = np.linspace(0, 24, 3, endpoint=False)
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
    dnoisy, noise_covar = SM.add_noise_uniform(d, 0.01)
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")


    # Set up the foreground model
    mod = FM.generate_binwise_cm21_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    p0 += cm21_params
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0, sigma=np.sqrt(noise_covar.diag), method="dogbox")

    # Compute chi-square.
    residuals = dnoisy.vector - mod(res[0])
    chi_sq = residuals @ noise_covar.matrix @ residuals
    chi_sq = np.sum(residuals**2/noise_covar.diag)
    print(f"Reduced chi square = {chi_sq/(len(dnoisy.vector)-(Npoly+3))}")

    if mcmc:
        # create a small ball around the MLE the initialize each walker
        nwalkers, fg_dim = 64, Npoly+3
        ndim = fg_dim
        pos = res[0]*(1 + 1e-4*np.random.randn(nwalkers, ndim))

        # run emcee without priors
        err = np.sqrt(noise_covar.diag)
        sampler = EnsembleSampler(nwalkers, ndim, NRI.log_likelihood, 
                            args=(dnoisy.vector, err, mod))
        _=sampler.run_mcmc(pos, nsteps=3000, progress=True)
        chain_mcmc = sampler.get_chain(flat=True, discard=1000)
        theta_inferred = np.mean(chain_mcmc, axis=0)
        c = ChainConsumer()
        c.add_chain(chain_mcmc)
        f = c.plotter.plot()
        plt.show()
    
    print("par est:", res[0])
    print("std devs:", np.sqrt(np.diag(res[1])))
    plt.errorbar(range(len(dnoisy.vector)), dnoisy.vector-mod(res[0]), yerr=np.sqrt(noise_covar.diag), fmt='.')
    plt.xlabel("bin")
    plt.ylabel("data - model [K]")
    plt.show()

    # Provide a corner plot for the 21-cm inference.
    # Draw samples from the likelihood.
    chain = np.random.multivariate_normal(mean=res[0][-3:], cov=res[1][-3:,-3:], size=100000)
    c = ChainConsumer()
    c.add_chain(chain, parameters=['A', 'nu0', 'dnu'])
    if mcmc:
        c.add_chain(chain_mcmc[:,-3:])
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

def fg_cm21_chrom(Npoly=3, mcmc=False, chrom=None):
    # Model and observation params
    nside   = 32
    lmax    = 32
    Nlmax   = RS.get_size(lmax)
    lats = [-26]#np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 3, endpoint=False)
    Ntau  = 1
    nuarr = np.linspace(50,100,51)
    cm21_params     = [-0.2, 80.0, 5.0]

    # Generate foreground and 21-cm alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm
    
    # Generate observation matrix for the modelling and for the observations.
    if chrom is None:
        chromfunc = BF.fwhm_func_tauscher
    else:
        chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau, lats, times, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)
    
    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise_uniform(d, 0.01)
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")

    # Set up the foreground model
    mod = FM.generate_binwise_cm21_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    p0 += cm21_params
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0, sigma=np.sqrt(noise_covar.diag))

    # Compute chi-square.
    residuals = dnoisy.vector - mod(res[0])
    chi_sq = residuals @ noise_covar.matrix @ residuals
    chi_sq = np.sum(residuals**2/noise_covar.diag)
    print(f"Reduced chi square = {chi_sq/(len(dnoisy.vector)-(Npoly+3))}")

    
    if mcmc:
        # create a small ball around the MLE to initialize each walker
        nwalkers, fg_dim = 64, Npoly+3
        ndim = fg_dim
        pos = res[0]*(1 + 1e-4*np.random.randn(nwalkers, ndim))

        # run emcee without priors
        err = np.sqrt(noise_covar.diag)
        sampler = EnsembleSampler(nwalkers, ndim, NRI.log_likelihood, 
                            args=(dnoisy.vector, err, mod))
        _=sampler.run_mcmc(pos, nsteps=3000, progress=True)
        chain_mcmc = sampler.get_chain(flat=True, discard=1000)
        theta_inferred = np.mean(chain_mcmc, axis=0)
        c = ChainConsumer()
        c.add_chain(chain_mcmc)
        f = c.plotter.plot()
        plt.show()

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
    if mcmc:
        c.add_chain(chain_mcmc[:,-3:])
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


def fg_cm21_chrom_corr(Npoly=3, mcmc=False, chrom=None, basemap_err=None, savetag=None, times=None, lats=None, mcmc_pos=None, steps=3000, burn_in=1000, fidmap_HS=False):
    """
    NOTE: will NOT work if Ntau != 1.

    basemap_err : fractional (not percentage) error on the T408 basemap, i.e.
                  of type 'bm'.
    """
    # Model and observation params
    nside   = 32
    lmax    = 32
    Nlmax   = RS.get_size(lmax)
    if lats is None:
        lats = np.array([-26*3, -26*2, -26, 0, 26, 26*2, 26*3])#np.linspace(-80, 80, 100)#[-26]#
    if times is None:
        times = np.linspace(0, 24, 12, endpoint=False)
    Ntau  = 1
    nuarr = np.linspace(50,100,51)
    cm21_params = OBS.cm21_params

    # Generate foreground and 21-cm alm
    if fidmap_HS:
        fg_alm = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, const_idx=True)
        print("Using Haslam-shifted foregrounds")
    else:
        fg_alm = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=SM.basemap_err_to_delta(basemap_err), err_type='idx')
        print("Using GSMA foregrounds with basemap error", basemap_err)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix for the observations.
    if chrom is None:
        # Generate observation matrix for the achromatic case.
        narrow_cosbeam = lambda x: BF.beam_cos(x, 0.8)
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, Ntau=Ntau, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=True)
        mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
        mat_G = BlockMatrix(mat=mat_G, mode='block', nblock=len(nuarr))
        mat_P = BlockMatrix(mat=mat_P, mode='block', nblock=len(nuarr))
        mat_Y = BlockMatrix(mat=mat_Y, mode='block', nblock=len(nuarr))
        mat_B = BlockMatrix(mat=mat_B, mode='block', nblock=len(nuarr))
    else:
        # Generate observation matrix for the chromatic case.
        chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau=Ntau, lats=lats, times=times, return_mat=True, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)

    # Perform fiducial observations without binning.
    d = mat_P @ mat_Y @ mat_B @ fid_alm
    np.save("saves/Binwise/mat_PYB.npy", (mat_P @ mat_Y @ mat_B).matrix)
    dnoisy, noise_covar = SM.add_noise(d, nuarr[1]-nuarr[0], len(times), t_int=200) # Ntau NOTE: using len(times) here because we keep the data unbinned to apply the chromaticity correction. Noise is recomputed later.
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")

    dnoisy_vector = dnoisy.vector
    if chrom is not None:
        # Perform an EDGES-style chromaticity correction.
        # Generate alm of the Haslam-shifted sky and observe them using our beam.
        has_alm = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, const_idx=True, delta=basemap_err, err_type='bm', seed=124)
        chrom_corr_numerator = mat_P @ mat_Y @ mat_B @ has_alm
        # Construct an observation matrix of the hypothetical (non-chromatic) case.
        mat_B_ref = BlockMatrix(mat=mat_B.block[10], nblock=mat_B.nblock)
        chrom_corr_denom = mat_P @ mat_Y @ mat_B_ref @ has_alm
        chrom_corr = chrom_corr_numerator.vector/chrom_corr_denom.vector
        np.save("saves/Binwise/chrom_corr.npy", chrom_corr)
        np.save("saves/Binwise/dnoisy_vector.npy", dnoisy_vector)
        np.save("saves/Binwise/noise.npy", np.sqrt(noise_covar.diag))
        dnoisy_vector /= chrom_corr
        np.save("saves/Binwise/dnoisy_vector_corr.npy", dnoisy_vector)
    elif chrom == 'bloop':
        # Perform an EDGES-style chromaticity correction.
        # Generate alm of the Haslam-shifted sky and observe them using our beam.
        has_alm = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, const_idx=True, delta=basemap_err, err_type='bm', seed=124)
        chrom_corr_numerator = mat_A @ has_alm
        # Construct an observation matrix of the hypothetical (non-chromatic) case.
        mat_A_ref = BlockMatrix(mat=mat_A.block[10], nblock=mat_A.nblock)
        chrom_corr_denom = mat_A_ref @ has_alm
        chrom_corr = chrom_corr_numerator.vector/chrom_corr_denom.vector
        dnoisy_vector /= chrom_corr

    # Bin the noisy data and the noise.
    dnoisy = mat_G@dnoisy_vector
    np.save("saves/Binwise/mat_G.npy", mat_G.matrix)


    noise = np.std([np.sqrt(np.diag(noise_covar.block[n])) for n in range(noise_covar.nblock)], axis=1)
    noise_covar_binned = np.diag(noise)**2
    #dnoisy = BlockVector(vec=dnoisy_vector, nblock=dnoisy.nblock)
    
    # Set up the foreground model
    mod = FM.genopt_binwise_cm21_forward_model(nuarr, mat_A, Npoly=Npoly)
    mod_prerun = FM.generate_binwise_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod_prerun(theta)

    # Try curve_fit, if it doesn't work just set res to the guess parameters.
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0, sigma=noise)
    print("par est:", res[0])
    print("std devs:", np.sqrt(np.diag(res[1])))
    
    '''# Compute chi-square.
    residuals = dnoisy.vector - mod(res[0])
    chi_sq = residuals @ noise_covar_binned @ residuals
    chi_sq = np.sum(residuals**2/np.diag(noise_covar_binned))
    print(f"Reduced chi square = {chi_sq/(len(dnoisy.vector)-(Npoly+3))}")'''

    if mcmc:
        # create a small ball around the MLE to initialize each walker
        nwalkers, fg_dim = 64, Npoly+3
        ndim = fg_dim
        if mcmc_pos is not None:
            if len(mcmc_pos) < ndim:
                print("error - mcmc start pos has too few params. Adding some.")
                params_to_add = [0.01]*(ndim-len(mcmc_pos))
                mcmc_pos = np.append(mcmc_pos[:-3], params_to_add)
                mcmc_pos = np.append(mcmc_pos, cm21_params)
            elif len(mcmc_pos) > ndim:
                print("error - mcmc start pos has too many params. Cutting some.")
                new_pos = mcmc_pos[:Npoly]
                new_pos = np.append(new_pos, cm21_params)
                mcmc_pos = np.array(new_pos)
            p0 = mcmc_pos
        else:
            p0 = np.append(res[0], OBS.cm21_params)
        priors = [[1, 25], [-3.5, -1.5]]
        priors += [[-10, 10.1]]*(Npoly-2)
        priors += [[-2, -0.001], [60, 90], [5, 15]]
        priors = np.array(priors)
        print("MCMC PRIORS:", priors)
        
        p0 = INF.prior_checker(priors, p0)
        pos = p0*(1 + 1e-4*np.random.randn(nwalkers, ndim))
        err = noise
        sampler = EnsembleSampler(nwalkers, ndim, NRI.log_posterior, 
                            args=(dnoisy.vector, err, mod, priors))
        _=sampler.run_mcmc(pos, nsteps=steps, progress=True, skip_initial_state_check=True)
        chain_mcmc = sampler.get_chain(flat=True, discard=burn_in)

    # Calculate the BIC for MCMC.
    bic = None
    if mcmc:
        c = ChainConsumer()
        c.add_chain(chain_mcmc, statistics='max')
        analysis_dict = c.analysis.get_summary(squeeze=True)
        theta_max = np.array([val[1] for val in analysis_dict.values()])
        loglike = NRI.log_likelihood(theta_max, y=dnoisy.vector, yerr=err, model=mod)
        bic = len(theta_max)*np.log(len(dnoisy)) - 2*loglike
        print("bic is ", bic)

    if savetag is not None:
        prestr = f"Nant<{len(lats)}>_Npoly<{Npoly}>"
        if chrom is None:
            prestr += "_achrom"
        else:
            prestr += "_chrom<{:.1e}>".format(chrom)
        if basemap_err is not None:
            prestr += '_bm'+f"<{basemap_err}>"
    
        np.save("saves/Binwise/"+prestr+savetag+"_data.npy", dnoisy.vector)
        np.save("saves/Binwise/"+prestr+savetag+"_dataerr.npy", noise)
        if mcmc:
            np.save("saves/Binwise/"+prestr+savetag+"_mcmcChain.npy", chain_mcmc)
        if bic is not None:
            print("saving bic")
            np.save("saves/Binwise/"+prestr+savetag+"_bic.npy", bic)

'''
def fg_cm21_chrom_corr_polych(Npoly=3, mcmc=False, chrom=None, basemap_err=0):
    """
    NOTE: will NOT work if Ntau != 1.
    This is the polychord version of the same function.
    """
    # Model and observation params
    nside   = 32
    lmax    = 32
    Nlmax   = RS.get_size(lmax)
    lats = [-26]#np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 3, endpoint=False)
    Ntau  = 1
    nuarr = np.linspace(50,100,51)
    cm21_params     = [-0.2, 80.0, 5.0]
    delta = SM.basemap_err_to_delta(percent_err=basemap_err)

    # Generate foreground and 21-cm alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=delta)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm
    
    # Generate observation matrix for the modelling and for the observations.
    if chrom is None:
        chromfunc = BF.fwhm_func_tauscher
    else:
        chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, Ntau, lats, times, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)

    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise_uniform(d, 0.01)
    err = np.sqrt(noise_covar.diag)
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")

    # Perform an EDGES-style chromaticity correction.
    # Generate alm of the Haslam-shifted sky and observe them using our beam.
    has_alm = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, const_idx=True)
    chrom_corr_numerator = mat_A @ has_alm
    # Construct an observation matrix of the hypothetical (non-chromatic) case.
    mat_A_ref = BlockMatrix(mat=mat_A.block[10], nblock=mat_A.nblock)
    chrom_corr_denom = mat_A_ref @ has_alm
    
    chrom_corr = chrom_corr_numerator.vector/chrom_corr_denom.vector
    dnoisy = BlockVector(vec=dnoisy.vector/chrom_corr, nblock=dnoisy.nblock)
    
    # Set up the foreground model
    mod = FM.generate_binwise_cm21_forward_model(nuarr, mat_A, Npoly=Npoly)
    def mod_cf(nuarr, *theta):
        theta = np.array(theta)
        return mod(theta)

    # Try curve_fit:
    p0 = [10, -2.5]
    p0 += [0.01]*(Npoly-2)
    p0 += cm21_params
    res = curve_fit(mod_cf, nuarr, dnoisy.vector, p0=p0, sigma=np.sqrt(noise_covar.diag))

    # Compute chi-square.
    residuals = dnoisy.vector - mod(res[0])
    chi_sq = residuals @ noise_covar.matrix @ residuals
    chi_sq = np.sum(residuals**2/noise_covar.diag)
    print(f"Reduced chi square = {chi_sq/(len(dnoisy.vector)-(Npoly+3))}")


    # Set up polychord.
    priors = [[1, 25], [-3.5, -1.5]]
    priors += [[-2, 2]]*(Npoly-2)
    priors += [[-0.5, -0.01], [60, 90], [1, 8]]
    priors = np.array(priors)
    nDims = Npoly+3
    polych_prior = INF.get_polychord_prior(prior_pars=priors)
    polych_likel = INF.get_polychord_loglikelihood(y=dnoisy.vector, yerr=err, model=mod)
    paramnames = [(f'p{i}', f'\\theta_{i}') for i in range(nDims)]
    output = pypolychord.run(
        polych_likel,
        nDims,
        nDerived=0,
        prior=polych_prior,
        dumper=INF.dumper,
        file_root='test',
        nlive=200,
        do_clustering=True,
        read_resume=False,
        paramnames=paramnames,
    )

    from anesthetic import make_2d_axes
    fig, ax = make_2d_axes(['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])
    output.plot_2d(ax)
    fig.savefig("AAAAAAplot.pdf")
    plt.show()
    '''
'''
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
    if mcmc:
        c.add_chain(chain_mcmc[:,-3:])
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
    '''
'''
if __name__ == "__main__":
    from mpi4py import MPI
    fg_cm21_chrom_corr_polych(Npoly=6, chrom=8e-3)
'''