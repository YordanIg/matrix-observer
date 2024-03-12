import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd
import pandas as pd
import seaborn as sns
from emcee import EnsembleSampler

from pygdsm import GlobalSkyModel2016

import src.forward_model as FM
import src.beam_functions as BF
import src.powerlaw_regression as PR
import src.sky_models as SM
import src.spherical_harmonics as SH
from src.blockmat import BlockMatrix, BlockVector
from src.spherical_harmonics import RealSphericalHarmonics
RS = RealSphericalHarmonics()

def fiducial_obs(uniform_noise=False):
    # Forward model the fiducial degraded GSMA.
    Nfreq = 51
    nuarr = np.linspace(50,100,Nfreq)
    lmax = 32
    nside = 16
    npix = hp.nside2npix(nside)
    narrow_cosbeam = lambda x: BF.beam_cos(x, theta0=0.8)
    fg_alm = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax)

    times = np.linspace(0,24,24, endpoint=False)
    mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr, nside, lmax, Ntau=len(times), times=times, beam_use=narrow_cosbeam, return_mat=True)

    d = mat_A@fg_alm
    if uniform_noise:
        dnoisy, noise_covar = SM.add_noise_uniform(temps=d, err=1)
    elif not uniform_noise:
        dnoisy, noise_covar = SM.add_noise(temps=d, dnu=1, Ntau=npix, t_int=1)
    return dnoisy, noise_covar, mat_A, mat_Y, nuarr


def mask_split(Nregions=9, visualise=False):
    # Split the sky into the Nregions.
    indx = np.load("anstey/indexes_16.npy")
    if visualise:
        hp.mollview(indx[0], title="GSMA lo 408 MHz amplitudes")
        plt.show()
        hp.mollview(indx[1], title="GSMA lo indexes")
        plt.show()

    max_indx = np.max(indx[1])+0.01
    min_indx = np.min(indx[1])-0.01
    indx_range = np.linspace(min_indx, max_indx, Nregions+1)
    inference_bounds = [(indx_range[i],indx_range[i+1]) for i in range(Nregions)]
    inference_bounds = np.array(inference_bounds)
    print(indx_range)
    print(inference_bounds)

    masks = []
    for i in range(len(indx_range)-1):
        mask = []
        range_tuple = (indx_range[i], indx_range[i+1])
        for j in range(len(indx[1])):
            if indx[1,j] > range_tuple[0] and indx[1,j] < range_tuple[1]:
                mask.append(j)
        masks.append(mask)
    mask_maps = []
    for mask in masks:
        mask_map = np.zeros(len(indx[1]))
        mask_map[mask] = 1.
        mask_maps.append(mask_map)
    
    if visualise:
        mask_visualisation = np.sum([(i+1)*mask_maps[i] for i in range(Nregions)], axis=0)
        hp.mollview(mask_visualisation, title=r"N$_\mathrm{regions}$="+f"{Nregions} map")
        plt.show()
    return mask_maps, inference_bounds



def log_likelihood(theta, y, yerr, model):
    """
    Compute the Gaussian log-likelihood, given a model(theta) and data y
    with errors yerr.
    """
    y_model = model(theta)
    chi2 = (y - y_model)**2 / (yerr**2)
    return np.sum(-chi2 / 2)

def log_prior(theta, prior_range):
    """
    Log Prior probability

        log(prior(theta))

        This handles an unnormalised uniform prior
        within the rectangular bounds given by prior_range.

        inputs:
        theta - N array of parameter values
        prior_range - [N,2] array of prior ranges
        i.e. = [[lower1, upper1], ...]

        Returns 0 if theta in prior_range and -inf otherwise
    """

    lower =  theta > prior_range[:,0]
    upper = theta < prior_range[:,1]
    in_prior_range = all(lower & upper)

    #return prior value
    if in_prior_range:
        return 0.0
    return -np.inf

def log_posterior(theta, y, yerr, model, prior_range):
    lp = log_prior(theta, prior_range)
    if np.isfinite(lp):
        lp += log_likelihood(theta, y, yerr, model)
    return lp


def inference(inference_bounds, noise_covar, dnoisy, model, steps=10000, theta_guess=None, tag=''):
    # create a small ball around the MLE the initialize each walker
    nwalkers, ndim = 32, len(inference_bounds)
    if theta_guess is None:
        theta_guess = np.array([0.5*(bound[0]+bound[1]) for bound in inference_bounds])
    pos = theta_guess*(1 + 1e-4*np.random.randn(nwalkers, ndim))
    priors = np.array([[-0.1, 5.0]*ndim])
    print("theta guess = {}".format(theta_guess))
    # run emcee
    err = np.sqrt(noise_covar.diag)
    sampler = EnsembleSampler(nwalkers, ndim, log_posterior, 
                        args=(dnoisy.vector, err, model, priors))
    _=sampler.run_mcmc(pos, steps, progress=True)
    np.save(f"saves/chain_anstey{ndim}regions_gsmalo_speedy{tag}", sampler.get_chain())  # SAVE THE CHAIN.


def main(Nregions=6, steps=10000, return_model=False, uniform_noise=True):
    """
    Run Nregions inference on observations of the degraded GSMA, with either 
    uniform or radiometric noise.
    """
    dnoisy, noise_covar, mat_A, mat_Y, nuarr = fiducial_obs(uniform_noise=uniform_noise)
    mask_maps, inference_bounds = mask_split(Nregions=Nregions)
    model = FM.genopt_nregions_pl_forward_model(nuarr=nuarr, masks=mask_maps, observation_mat=mat_A, spherical_harmonic_mat=mat_Y)
    if return_model:
        return model
    model(theta=np.array([2]*Nregions))
    inference(inference_bounds, noise_covar, dnoisy, model, steps=steps)


def main_tworun(Nregions=9, steps=10000, uniform_noise=True):
    """
    Do the same as main, but run inference with larger errors, then with smaller
    errors, starting at the mean inferred parameter position of the prior run.
    """
    dnoisy, noise_covar, mat_A, mat_Y, nuarr = fiducial_obs(uniform_noise=uniform_noise)
    mask_maps, inference_bounds = mask_split(Nregions=Nregions)
    model = FM.genopt_nregions_pl_forward_model(nuarr=nuarr, masks=mask_maps, observation_mat=mat_A, spherical_harmonic_mat=mat_Y)
    model(theta=np.array([2]*Nregions))
    inference(inference_bounds, noise_covar*100, dnoisy, model, steps=steps, tag='_0')

    chain = np.load(f"saves/chain_anstey{Nregions}regions_gsmalo_speedy_0.npy")
    chain = chain[15000:]  # Burn-in.
    ch_sh = np.shape(chain)
    chain_flat = np.reshape(chain, (ch_sh[0]*ch_sh[1], ch_sh[2]))  # Flatten chain.
    theta_guess = np.mean(chain_flat, axis=0)
    inference(inference_bounds, noise_covar, dnoisy, model, steps=steps, theta_guess=theta_guess, tag='_1')


def _test_forward_models():
    """
    Check to see that the forward modelling is producing the same results for 
    both implementations.
    """
    dnoisy, noise_covar, mat_A, mat_Y, nuarr = fiducial_obs()
    mask_maps, inference_bounds = mask_split(Nregions=9)
    model = FM.generate_nregions_pl_forward_model(nuarr=nuarr, masks=mask_maps, observation_mat=mat_A, spherical_harmonic_mat=mat_Y)
    model_opt = FM.genopt_nregions_pl_forward_model(nuarr=nuarr, masks=mask_maps, observation_mat=mat_A, spherical_harmonic_mat=mat_Y)
    m1 = model_opt(theta=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]))
    m2 = model(theta=np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]))
    plt.plot(m1 - m2, '.')
    plt.show()
