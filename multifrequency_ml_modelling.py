"""
Using maximum-likelihood methods to reconstruct a_{00}(\nu), then fit a power
law and a 21-cm signal to it.
"""
from functools import partial
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from scipy.optimize import curve_fit
from emcee import EnsembleSampler

import src.beam_functions as BF
import src.spherical_harmonics as SH
import src.forward_model as FM
import src.sky_models as SM
import src.map_making as MM
import src.plotting as PL
from src.blockmat import BlockMatrix, BlockVector
import src.inference as INF
from anstey.generate import T_CMB

RS = SH.RealSphericalHarmonics()

# Fit the foreground and 21-cm monopole.
def fg_polymod(nuarr, *theta_fg):
    Afg, alpha = theta_fg[:2]
    zetas      = theta_fg[2:]
    exponent = [zetas[i]*np.log(nuarr/60)**(i+2) for i in range(len(zetas))]
    fg_a00_terms = (Afg*1e3)*(nuarr/60)**(-alpha) * np.exp(np.sum(exponent, 0))
    return fg_a00_terms + np.sqrt(4*np.pi)*T_CMB

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
    if lmax != lmod:
        mat_A_mod = FM.calc_observation_matrix_all_pix(nside, lmod, Ntau=npix, beam_use=narrow_cosbeam)
        mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))
    else:
        mat_A_mod = mat_A

    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=npix, t_int=1e4)

    # Reconstruct the max likelihood estimate of the alm
    _ = MM.calc_ml_estimator_matrix(mat_A=mat_A_mod.block[-1], mat_N=noise_covar.block[-1], cond=True)
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
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=1e4)

    # Reconstruct the max likelihood estimate of the alm
    mat_W, cov = MM.calc_ml_estimator_matrix(mat_A_mod, noise_covar, cov=True)
    alm_error = np.sqrt(cov.diag)
    rec_alm = mat_W @ dnoisy

    # Extract the monopole component of the reconstructed alm.
    fg_a00  = np.array(fg_alm[::Nlmax])
    rec_a00 = np.array(rec_alm.vector[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    fg_mon_p0 = [15, 2.5, .001]
    cm21_mon_p0 = [-0.2, 80, 5]
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, p0=fg_mon_p0+cm21_mon_p0)
    
    _plot_results(nuarr, Nlmax, Nlmod, rec_alm, alm_error, fid_alm, cm21_alm, res)


def nontrivial_obs_ndarrays():
    # Model and observation params
    nside   = 32
    lmax    = 32
    lmod    = 8
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)
    npix    = hp.nside2npix(nside)
    delta   = 1e-2
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
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=1e4)

    # Reconstruct the max likelihood estimate of the alm
    mat_W, cov  = MM.calc_ml_estimator_matrix(mat_A_mod, noise_covar, cov=True, delta=delta, reg='exp', nuarr=nuarr, pow=2.5)
    alm_error = np.sqrt(np.diag(cov))
    rec_alm = mat_W @ dnoisy.vector

    '''
    # Compute the chi-square and compare it to the length of the data vector.
    chi_sq = (dnoisy - mat_A_mod@rec_alm).T @ noise_covar.inv.matrix @ (dnoisy - mat_A_mod@rec_alm)
    chi_sq = sum(chi_sq.diag)
    print("Chi-square:", chi_sq, "len(data):", dnoisy.vec_len,"+/-", np.sqrt(2*dnoisy.vec_len), "Nparams:", Nlmod*len(nuarr))
    '''
    # Extract the monopole component of the reconstructed alm.
    rec_a00 = np.array(rec_alm[::Nlmod])
    a00_error = np.array(alm_error[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    Npoly = 3
    fg_mon_p0 = [15, 2.5]
    fg_mon_p0 += [.001]*(Npoly-2)
    cm21_mon_p0 = [-0.2, 80, 5]
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, sigma=a00_error, p0=fg_mon_p0+cm21_mon_p0)
    
    _plot_results(nuarr, Nlmax, Nlmod, rec_alm, alm_error, fid_alm, cm21_alm, res)


def nontrivial_obs_memopt(chrom=None, missing_modes=False, basemap_err=0, reg_delta=None):
    """
    A memory-friendly version of nontrivial_obs which computes the reconstruction
    of each frequency seperately, then brings them all together.
    """
    # Model and observation params
    nside   = 32
    lmax    = 32
    lmod    = 3
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)
    lats = np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 12, endpoint=False)#np.linspace(0, 24, 144, endpoint=False)  # 144 = 10 mins per readout
    nuarr = np.linspace(50,100,51)
    cm21_params     = (-0.2, 80.0, 5.0)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground and 21-cm signal alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=SM.basemap_err_to_delta(percent_err=basemap_err))
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix for the modelling and for the observations.
    if chrom is not None:
        if not isinstance(chrom, bool):
            chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
        else:
            chromfunc = BF.fwhm_func_tauscher
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, lats=lats, times=times, return_mat=True, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)
        mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmod, lats=lats, times=times, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)
    elif chrom is None:
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=True)
        mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
        mat_G = BlockMatrix(mat=mat_G, mode='block', nblock=len(nuarr))
        mat_P = BlockMatrix(mat=mat_P, mode='block', nblock=len(nuarr))
        mat_B = BlockMatrix(mat=mat_B, mode='block', nblock=len(nuarr))
        mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmod, lats=lats, times=times, beam_use=narrow_cosbeam)
        mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))
            
    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=100, seed=456)#t_int=100, seed=456)#
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")

    # Optionally generate a missing-modes correction.
    if missing_modes:
        if lmax==lmod:
            print("missing modes correction will not be computed - lmax=lmod")
        elif lmax > lmod:
            mat_S = MM.calc_nongauss_unmodelled_mode_matrix(lmod=lmod, alm_vector=fg_alm, mat_A_blocks=mat_A.block)
            noise_covar += mat_S
    
    # Reconstruct the max likelihood estimate of the alm
    if reg_delta is None:                                                       # will keep everything as blockvectors
        mat_W, cov = MM.calc_ml_estimator_matrix(mat_A=mat_A_mod, mat_N=noise_covar, cov=True)
        alm_error = np.sqrt(cov.diag)
        rec_alm = mat_W @ dnoisy
        # Compute the chi-square and compare it to the length of the data vector.
        chi_sq = (dnoisy - mat_A_mod@rec_alm).T @ noise_covar.inv @ (dnoisy - mat_A_mod@rec_alm)
        chi_sq = sum(chi_sq.diag)
        print("Chi-square:", chi_sq, "len(data):", dnoisy.vec_len,"+/-", np.sqrt(2*dnoisy.vec_len), "Nparams:", Nlmod*len(nuarr))
        
        rec_a00 = np.array(rec_alm.vector[::Nlmod])
    
    elif reg_delta is not None:                                                 # will convert everything to numpy arrays
        mat_W, cov = MM.calc_ml_estimator_matrix(mat_A=mat_A_mod, mat_N=noise_covar, cov=True, reg='exp', nuarr=nuarr, pow=-2.5, delta=reg_delta)
        alm_error = np.sqrt(np.diag(cov))
        rec_alm = mat_W @ dnoisy.vector
        rec_a00 = np.array(rec_alm[::Nlmod])

 

    # Extract the monopole component of the reconstructed alm.
    a00_error = np.array(alm_error[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    Npoly = 8
    fg_mon_p0 = [15, 2.5]
    fg_mon_p0 += [.001]*(Npoly-2)
    cm21_mon_p0 = [-0.2, 80, 5]
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, sigma=a00_error, p0=fg_mon_p0+cm21_mon_p0)

    if reg_delta is None:
        _plot_results(nuarr, Nlmax, Nlmod, rec_alm.vector, alm_error, fid_alm, cm21_alm, res)
    elif reg_delta is not None:
        _plot_results(nuarr, Nlmax, Nlmod, rec_alm, alm_error, fid_alm, cm21_alm, res)


def nontrivial_obs_memopt_missing_modes(Npoly=9, chrom=None, basemap_err=0.05, err_type='idx', mcmc=False, mcmc_pos=None, savetag=""):
    """
    A memory-friendly version of nontrivial_obs which computes the reconstruction
    of each frequency seperately, then brings them all together.
    """
    # Model and observation params
    nside   = 32
    lmax    = 32
    lmod    = 3
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)
    lats = np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 12, endpoint=False)#np.linspace(0, 24, 144, endpoint=False)  # 144 = 10 mins per readout
    nuarr = np.linspace(50,100,51)
    cm21_params     = (-0.2, 80.0, 5.0)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground and 21-cm signal alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=SM.basemap_err_to_delta(basemap_err), err_type=err_type, seed=100, meancorr=False)
    cm21_alm = SM.cm21_gauss_mon_alm(nu=nuarr, lmax=lmax, params=cm21_params)
    fid_alm  = fg_alm + cm21_alm

    # Generate observation matrix for the modelling and for the observations.
    if chrom is not None:
        if not isinstance(chrom, bool):
            chromfunc = partial(BF.fwhm_func_tauscher, c=chrom)
        else:
            chromfunc = BF.fwhm_func_tauscher
        mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmax, lats=lats, times=times, return_mat=False, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)
        mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(nuarr, nside, lmod, lats=lats, times=times, return_mat=False, beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)
    elif chrom is None:
        mat_A = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=False)
        mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
        mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmod, lats=lats, times=times, beam_use=narrow_cosbeam, return_mat=False)
        mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))
    
    # Perform fiducial observations
    d = mat_A @ fid_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=100, seed=456)#t_int=100, seed=456)#
    sample_noise = np.sqrt(noise_covar.block[0][0,0])
    print(f"Data generated with noise {sample_noise} K at 50 MHz in the first bin")

    # Generate a missing-modes correction.
    # Step 1: Generate instances of the GSMA and find the mean and covariance
    #         of unmodelled modes.
    fg_alm_list = []
    for i in range(10):
        if err_type=='idx':
            fg_alm_list.append(SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=SM.basemap_err_to_delta(basemap_err), err_type=err_type, seed=123+i, meancorr=False))
        else:
            fg_alm_list.append(SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True, delta=basemap_err, err_type=err_type, seed=123+i))
    fg_alm_arr = np.array(fg_alm_list)
    fg_alm_arr = np.array(np.split(fg_alm_arr, len(nuarr), axis=1))
    fg_alm_unmod_arr  = fg_alm_arr[:,:,Nlmod:]

    # Step 2: Calculate the missing-modes observation matrix.
    mat_A_unmod = BlockMatrix(mat_A.block[:,:,Nlmod:])

    # Step 3: Compute the data correction and covariance correction.
    data_corr  = []
    covar_corr = []
    for alm_block, mat_A_unmod_block in zip(fg_alm_unmod_arr, mat_A_unmod.block):
        alm_block_mean = np.mean(alm_block, axis=0)
        alm_block_cov  = np.cov(alm_block, rowvar=False)
        data_corr.append(mat_A_unmod_block @ alm_block_mean)
        covar_corr.append(mat_A_unmod_block @ alm_block_cov @ mat_A_unmod_block.T)
    data_corr = BlockVector(np.array(data_corr))
    covar_corr = BlockMatrix(np.array(covar_corr))

    # Reconstruct the max likelihood estimate of the alm
    mat_W, cov = MM.calc_ml_estimator_matrix(mat_A=mat_A_mod, mat_N=noise_covar+covar_corr, cov=True, cond=True)
    alm_error = np.sqrt(cov.diag)
    rec_alm = mat_W @ (dnoisy - data_corr)
    # Compute the chi-square and compare it to the length of the data vector.
    chi_sq = (dnoisy - mat_A_mod@rec_alm).T @ noise_covar.inv @ (dnoisy - mat_A_mod@rec_alm)
    chi_sq = sum(chi_sq.diag)
    print("Chi-square:", chi_sq, "len(data):", dnoisy.vec_len,"+/-", np.sqrt(2*dnoisy.vec_len), "Nparams:", Nlmod*len(nuarr))
    
    rec_a00 = np.array(rec_alm.vector[::Nlmod])
 

    # Extract the monopole component of the reconstructed alm.
    a00_error = np.array(alm_error[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial and 21-cm gaussian
    fg_mon_p0 = [15, 2.5]
    fg_mon_p0 += [.001]*(Npoly-2)
    cm21_mon_p0 = [-0.2, 80, 5]
    res = curve_fit(f=fg_cm21_polymod, xdata=nuarr, ydata=rec_a00, sigma=a00_error, p0=fg_mon_p0+cm21_mon_p0)

    _plot_results(nuarr, Nlmax, Nlmod, rec_alm.vector, alm_error, fid_alm, cm21_alm, res)

    if mcmc:
        def mod(theta):
            return fg_cm21_polymod(nuarr, *theta)
        
        # create a small ball around the MLE to initialize each walker
        nwalkers, fg_dim = 64, Npoly+3
        ndim = fg_dim
        if mcmc_pos is not None:
            pos = mcmc_pos*(1 + 1e-4*np.random.randn(nwalkers, ndim))
        else:
            pos = res[0]*(1 + 1e-4*np.random.randn(nwalkers, ndim))
        priors = [[1, 25], [1.5, 3.5]]
        priors += [[-2, 2.1]]*(Npoly-2)
        priors += [[-0.5, -0.01], [60, 90], [1, 8]]
        priors = np.array(priors)
        # run emcee without priors
        err = np.sqrt(noise_covar.diag)
        sampler = EnsembleSampler(nwalkers, ndim, INF.log_posterior, 
                            args=(rec_a00, a00_error, mod, priors))
        _=sampler.run_mcmc(pos, nsteps=3000, progress=True, skip_initial_state_check=True)
        chain_mcmc = sampler.get_chain(flat=True, discard=1000)
        theta_inferred = np.mean(chain_mcmc, axis=0)

        prestr = f"Nant<{len(lats)}>_Npoly<{Npoly}>_"
        if chrom is None:
            prestr += "achrom_"
        else:
            prestr += "chrom<{:.1e}>_".format(chrom)
        if basemap_err is not None:
            prestr += err_type+"<{}>_".format(basemap_err)
    
        np.save("saves/MLmod/"+prestr+savetag+"mcmcChain.npy", chain_mcmc)
        
        # Calculate the BIC for MCMC.    
        c = ChainConsumer()
        c.add_chain(chain_mcmc, statistics='max')
        analysis_dict = c.analysis.get_summary(squeeze=True)
        theta_max = np.array([val[1] for val in analysis_dict.values()])
        loglike = INF.log_likelihood(theta_max, y=rec_a00, yerr=a00_error, model=mod)
        bic = len(theta_max)*np.log(len(rec_a00)) - 2*loglike
        print("bic is ", bic)
        np.save("saves/MLmod/"+prestr+savetag+"bic.npy", bic)

        c = ChainConsumer()
        c.add_chain(chain_mcmc)
        f = c.plotter.plot()
        plt.show()

    

    del mat_A
    del mat_A_mod
    del mat_A_unmod


def nontrivial_fg_obs_memopt(ret=False):
    """
    A memory-friendly version of nontrivial_obs which computes the reconstruction
    of each frequency seperately, then brings them all together.

    This one only reconstructs foregrounds.
    """
    # Model and observation params
    nside   = 32
    lmax    = 32
    lmod    = 32
    delta   = 1e-3
    Nlmax   = RS.get_size(lmax)
    Nlmod   = RS.get_size(lmod)
    npix    = hp.nside2npix(nside)
    lats = np.array([-26*2, -26, 26, 26*2])#np.linspace(-80, 80, 100)#
    times = np.linspace(0, 24, 144, endpoint=False)
    nuarr   = np.linspace(50,100,51)
    narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)

    # Generate foreground alm
    fg_alm   = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax, nside=nside, use_mat_Y=True)

    # Generate observation matrix for the modelling and for the observations.
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmax, lats=lats, times=times, beam_use=narrow_cosbeam)
    mat_A = BlockMatrix(mat=mat_A, mode='block', nblock=len(nuarr))
    mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmod, lats=lats, times=times, beam_use=narrow_cosbeam)
    mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))
    
    # Perform fiducial observations
    d = mat_A @ fg_alm
    dnoisy, noise_covar = SM.add_noise(d, 1, Ntau=len(times), t_int=2500, seed=456)


    # Reconstruct the max likelihood estimate of the alm
    mat_Ws   = [MM.calc_ml_estimator_matrix(mat_A_mod_block, noise_covar_block, delta=delta) for mat_A_mod_block, noise_covar_block in zip(mat_A_mod.block, noise_covar.block)]
    mat_W = BlockMatrix(mat=np.array(mat_Ws))
    rec_alm = mat_W @ dnoisy

    # Extract the monopole component of the reconstructed alm.
    fg_a00  = np.array(fg_alm[::Nlmax])
    rec_a00 = np.array(rec_alm.vector[::Nlmod])

    # Fit the reconstructed a00 component with a polynomial
    fg_mon_p0 = [15, 2.5, .001]
    res = curve_fit(f=fg_polymod, xdata=nuarr, ydata=rec_a00, p0=fg_mon_p0)

    plt.plot(nuarr, rec_a00-fg_a00, label='$a_{00}$ reconstructed - $a_{00}$ fid fg')
    plt.axhline(y=0, linestyle=":", color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()
    plt.plot(nuarr, rec_a00-fg_polymod(nuarr, *res[0]), label='$a_{00}$ reconstructed - $a_{00}$ polyfit')
    plt.axhline(y=0, linestyle=":", color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()


def _plot_results(nuarr, Nlmax, Nlmod, rec_alm, alm_error, fid_alm, cm21_alm, final_fitres):
    fg_alm = fid_alm-cm21_alm

    # Extract the monopole component of the reconstructed alm.
    fid_a00  = np.array(fid_alm[::Nlmax])
    fg_a00  = np.array(fg_alm[::Nlmax])
    rec_a00 = np.array(rec_alm[::Nlmod])
    a00_error = np.array(alm_error[::Nlmod])

    # Plot the reconstructed a00 mode minus the fiducial a00 mode.
    plt.plot(nuarr, rec_a00-fid_a00, label='$a_{00}$ reconstructed - $a_{00}$ fid fg')
    plt.axhline(y=0, linestyle=":", color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()

    # Plot the reconstructed a00 mode minus the best-fitting power law with no
    # running.
    res = curve_fit(fg_polymod, xdata=nuarr, ydata=rec_a00, sigma=a00_error, p0=[15,2.5])
    plt.plot(nuarr, rec_a00-fg_polymod(nuarr, *res[0]), label='$a_{00}$ reconstructed - power law')
    plt.axhline(y=0, linestyle=":", color='k')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.show()

    # Provide a corner plot for the 21-cm inference.
    # Draw samples from the likelihood.
    chain = np.random.multivariate_normal(mean=final_fitres[0][-3:], cov=final_fitres[1][-3:,-3:], size=100000)
    c = ChainConsumer()
    c.add_chain(chain, parameters=['A', 'nu0', 'dnu'])
    f = c.plotter.plot()
    plt.show()

    # Evaluate the model at 100 points drawn from the chain to get 1sigma 
    # inference bounds in data space.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    chain_samples = np.random.multivariate_normal(mean=final_fitres[0][-3:], cov=final_fitres[1][-3:,-3:], size=100)
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
