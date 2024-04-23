"""
Run Nregions power law inference on the degraded GSMA sky plus the 21-cm 
monopole sky. Provides functions to do the forward modelling, inference and 
plotting of results. Saves chains in:
    saves/Nregs_pl_gsmalo_cm21mon/{Nregions}reg{noisetag}_(0/1).npy
where Nregions is the number of regions, noisetag can be either 'unoise' =
uniform noise or 'radnoise' = radiometric noise. 

main runs inference once, while main_tworun runs it twice, appending _0 and _1 
on the filenames. The first run uses larger errors than the second to zero in on
the correct posterior position.
"""
from pickle import dump
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from emcee import EnsembleSampler
from chainconsumer import ChainConsumer

import src.forward_model as FM
import src.beam_functions as BF
import src.sky_models as SM
from src.spherical_harmonics import RealSphericalHarmonics
RS = RealSphericalHarmonics()


# Fiducial frequency array
Nfreq = 51
nuarr = np.linspace(50,100,Nfreq)

# Default parameters of the observation and fiducial sky.
default_pars = {
    "times"  : np.linspace(0,24,3, endpoint=False),
    "unoise" : 1,                                         # Uniform noise level in Kelvin (only used if uniform_noise=True).
    "tint"   : 1,                                         # Total integration time in hours (only used if uniform_noise=False).
    "lmax"   : 32,
    "nside"  : 16
}

def fiducial_obs(uniform_noise=False, unoise_K=None, tint=None, times=None, 
                 lmax=None, nside=None):
    """
    Forward model the fiducial degraded GSMA.

    Parameters
    ----------
    uniform_noise
        If uniform_noise is True, add gaussian uniform noise with level unoise_K 
        kelvin. If false, add radiometric noise with tint hours of total
        integration time.
    times
        Observation times in hours. Defaults to default_pars["times"].
    lmax, nside
        Lmax and Nside of the fiducial sky. Defults to 
        default_pars["lmax/nside"].
    """
    if unoise_K is None:
        unoise_K = default_pars["unoise"]
    if tint is None:
        tint = default_pars["tint"]
    if times is None:
        times = default_pars["times"]
    if lmax is None:
        lmax = default_pars["lmax"]
    if nside is None:
        nside = default_pars["nside"]

    narrow_cosbeam = lambda x: BF.beam_cos(x, theta0=0.8)
    fg_alm = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmax)

    times = np.linspace(0,24,3, endpoint=False)
    mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr, nside, lmax, Ntau=len(times), times=times, beam_use=narrow_cosbeam, return_mat=True)

    d = mat_A@(fg_alm)
    if uniform_noise:
        dnoisy, noise_covar = SM.add_noise_uniform(temps=d, err=unoise_K)
    elif not uniform_noise:
        dnoisy, noise_covar = SM.add_noise(temps=d, dnu=1, Ntau=len(times), t_int=tint)

    params = {
        "uniform_noise" : uniform_noise,
        "unoise" : unoise_K,
        "tint"   : tint,
        "times"  : times,
        "lmax"   : lmax,
        "nside"  : nside
    }
    return dnoisy, noise_covar, mat_A, mat_Y, params


def mask_split(Nregions=9, visualise=False, nside=None):
    """
    Split the degraded GSMA sky into Nregions, returning as a tuple:
        (mask_maps, inference_bounds)
    where the inference bounds are the power law boundaries of each mask. Note 
    that this is a bit of a misnomer: often the inferred parameters will
    actually not be bounded by these inference bounds.
    """
    if nside is None:
        nside = default_pars['nside']
    # Split the sky into the Nregions.
    try:
        indx = np.load(f"anstey/indexes_{nside}.npy")
    except:
        raise ValueError(f"indexes for GSMA with nside={nside} have not been generated.")
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


def inference(inference_bounds, noise_covar, dnoisy, model, steps=10000, 
              theta_fg_guess=None, tag=''):
    """
    Run inference.
    If inference bounds are passed, they will be bisected to find a guess at the 
    starting inference position. If theta_guess is passed, this information is 
    disregarded.

    Parameters
    ----------
    inference_bounds
        The bounds of each power law region. NOT used as priors.
    noise_covar
        Noise covariance matrix
    dnoisy
        Noisy data
    model
        The inference model.
    steps
        Number of steps to take in the inference.
    theta_fg_guess
        Initial guess of the foreground parameters. Defults to the middle value
        of inference_bounds.
    tag
        Optional tag to add to the chain that's being saved. 
    """
    # create a small ball around the MLE the initialize each walker
    nwalkers, fg_dim = 32, len(inference_bounds)
    ndim = fg_dim

    if theta_fg_guess is None:
        theta_guess = [0.5*(bound[0]+bound[1]) for bound in inference_bounds]
        theta_guess = np.array(theta_guess)
    elif theta_fg_guess is not None:
        theta_guess = np.array(theta_fg_guess)
    pos = theta_guess*(1 + 1e-4*np.random.randn(nwalkers, ndim))

    priors = [[0.5, 5.0]]*fg_dim
    priors = np.array(priors)
    # run emcee
    err = np.sqrt(noise_covar.diag)
    sampler = EnsembleSampler(nwalkers, ndim, log_posterior, 
                        args=(dnoisy.vector, err, model, priors))
    _=sampler.run_mcmc(pos, steps, progress=True)
    np.save(f"saves/Nregs_pl_gsmalo/{fg_dim}reg{tag}", sampler.get_chain())  # SAVE THE CHAIN.


def main(Nregions=6, steps=10000, return_model=False, uniform_noise=True, tag="", 
        unoise_K=None, tint=None, times=None, lmax=None, nside=None, 
        theta_fg_guess=None):
    """
    Run Nregions inference on observations of the degraded GSMA, with either 
    uniform or radiometric noise.
    """
    if uniform_noise:
        noisetag = '_unoise'
    elif not uniform_noise:
        noisetag = '_radnoise'

    dnoisy, noise_covar, mat_A, mat_Y, pars = fiducial_obs(uniform_noise, unoise_K, tint, times, lmax, nside)
    np.save(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_data.npy", dnoisy.vector)
    with open(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_pars.pkl", "wb") as f:
        dump(pars, f)

    mask_maps, inference_bounds = mask_split(Nregions=Nregions, nside=nside)
    model = FM.genopt_nregions_pl_forward_model(nuarr=nuarr, masks=mask_maps, observation_mat=mat_A, spherical_harmonic_mat=mat_Y)
    if return_model:
        return model
    
    inference(inference_bounds, noise_covar, dnoisy, model, steps=steps, theta_fg_guess=theta_fg_guess, tag=f'{noisetag}{tag}')


def main_tworun(Nregions=6, steps=10000, uniform_noise=True, tag="", 
        unoise_K=None, tint=None, times=None, lmax=None, nside=None, 
        theta_fg_guess=None):
    """
    Do the same as main, but run inference with larger errors, then with smaller
    errors, starting at the mean inferred parameter position of the prior run.
    """
    if uniform_noise:
        noisetag = '_unoise'
    elif not uniform_noise:
        noisetag = '_radnoise'

    dnoisy, noise_covar, mat_A, mat_Y, pars = fiducial_obs(uniform_noise, unoise_K, tint, times, lmax, nside)
    np.save(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_data.npy", dnoisy.vector)
    with open(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_pars.pkl", "wb") as f:
        dump(pars, f)

    mask_maps, inference_bounds = mask_split(Nregions=Nregions, nside=nside)
    model = FM.genopt_nregions_pl_forward_model(nuarr=nuarr, masks=mask_maps, observation_mat=mat_A, spherical_harmonic_mat=mat_Y)

    # Run inference the first time.
    inference(inference_bounds, noise_covar*100, dnoisy, model, steps=steps, theta_fg_guess=theta_fg_guess, tag=f'{noisetag}{tag}_0')

    chain = np.load(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_0.npy")
    chain = chain[5000:]  # Burn-in.
    ch_sh = np.shape(chain)
    chain_flat = np.reshape(chain, (ch_sh[0]*ch_sh[1], ch_sh[2]))  # Flatten chain.
    theta_guess = np.mean(chain_flat, axis=0)
    # Run inference the second time.
    inference(inference_bounds, noise_covar, dnoisy, model, steps=steps, theta_fg_guess=theta_guess, tag=f'{noisetag}{tag}_1')


def main_threerun(Nregions=10, pre_steps=20000, steps=100000, uniform_noise=True, tag="", 
        unoise_K=None, tint=None, times=None, lmax=None, nside=None, 
        theta_fg_guess=None):
    """
    Do the same as main, but run inference with larger errors, then with smaller
    errors, starting at the mean inferred parameter position of the prior run.
    """
    if uniform_noise:
        noisetag = '_unoise'
    elif not uniform_noise:
        noisetag = '_radnoise'

    dnoisy, noise_covar, mat_A, mat_Y, pars = fiducial_obs(uniform_noise, unoise_K, tint, times, lmax, nside)
    np.save(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_data.npy", dnoisy.vector)
    with open(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_pars.pkl", "wb") as f:
        dump(pars, f)

    mask_maps, inference_bounds = mask_split(Nregions=Nregions, nside=nside)
    model = FM.genopt_nregions_pl_forward_model(nuarr=nuarr, masks=mask_maps, observation_mat=mat_A, spherical_harmonic_mat=mat_Y)

    # Run inference the first time.
    inference(inference_bounds, noise_covar*100, dnoisy, model, steps=pre_steps, theta_fg_guess=theta_fg_guess, tag=f'{noisetag}{tag}_0')

    chain = np.load(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_0.npy")
    chain = chain[15000:]  # Burn-in.
    ch_sh = np.shape(chain)
    chain_flat = np.reshape(chain, (ch_sh[0]*ch_sh[1], ch_sh[2]))  # Flatten chain.
    theta_guess = np.mean(chain_flat, axis=0)
    # Run inference the second time.
    inference(inference_bounds, noise_covar, dnoisy, model, steps=pre_steps, theta_fg_guess=theta_guess, tag=f'{noisetag}{tag}_1')

    chain = np.load(f"saves/Nregs_pl_gsmalo/{Nregions}reg{noisetag}{tag}_1.npy")
    chain = chain[15000:]  # Burn-in.
    ch_sh = np.shape(chain)
    chain_flat = np.reshape(chain, (ch_sh[0]*ch_sh[1], ch_sh[2]))  # Flatten chain.
    c = ChainConsumer()
    c.add_chain(chain_flat)
    analysis_dict = c.analysis.get_summary(squeeze=True)
    theta_guess = np.array([val[1] for val in analysis_dict.values()])
    # Run inference the third time.
    inference(inference_bounds, noise_covar, dnoisy, model, steps=steps, theta_fg_guess=theta_guess, tag=f'{noisetag}{tag}_2')


def plot_non_uniform_noise_comparison():
    """
    Looks like the radiometric noise inference produces different residuals and
    inferred parameters to the uniform noise case. This is likely because the 
    radiometric case places less relative importance on the noisier, 
    lower-frequency values.
    """
    from chainconsumer import ChainConsumer

    _, inference_bounds = mask_split(Nregions=9)

    # Radiometric noise.
    chain = np.load("saves/Nregs_pl_gsmalo/9reg_radnoise_1.npy")
    c=ChainConsumer()
    chain = chain[10000:]  # Burn-in.
    ch_sh = np.shape(chain)
    chain_flat = np.reshape(chain, (ch_sh[0]*ch_sh[1], ch_sh[2]))  # Flatten chain.
    '''c.add_chain(chain_flat, parameters=[r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_3$',r'$\gamma_4$',r'$\gamma_5$',r'$\gamma_6$',r'$\gamma_7$',r'$\gamma_8$',r'$\gamma_9$'])
    c.plotter.plot()
    plt.show()'''

    # Uniform noise.
    chain = np.load("saves/Nregs_pl_gsmalo/9reg_unoise_1.npy")
    c=ChainConsumer()
    chain = chain[10000:]  # Burn-in.
    ch_sh = np.shape(chain)
    chain_flat_unoise = np.reshape(chain, (ch_sh[0]*ch_sh[1], ch_sh[2]))  # Flatten chain.
    '''c.add_chain(chain_flat_unoise, parameters=[r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_3$',r'$\gamma_4$',r'$\gamma_5$',r'$\gamma_6$',r'$\gamma_7$',r'$\gamma_8$',r'$\gamma_9$'])
    c.plotter.plot()
    plt.show()'''
    
    # Generate most-likely data.
    theta_mean = np.mean(chain_flat, axis=0)
    theta_mean_unoise = np.mean(chain_flat_unoise, axis=0)

    model = main(Nregions=9, return_model=True)
    dnoisy, _, _, _, _ = fiducial_obs(uniform_noise=False)

    model_temps = model(theta=theta_mean)
    model_temps_unoise = model(theta=theta_mean_unoise)

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(dnoisy.vector, '.', label='data - radiometric noise')
    ax[0].plot(model_temps, '.', label='model')
    ax[1].plot(dnoisy.vector-model_temps, '.')
    ax[0].set_xlabel('bin')
    ax[1].set_xlabel('bin')
    ax[0].set_ylabel('Temperature [K]')
    ax[1].set_ylabel('Temperature residuals [K]')
    ax[0].legend()
    fig.tight_layout()
    fig.savefig('fig/non_uniform_noise_comparison_0.png')

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].plot(dnoisy.vector, '.', label='data - uniform noise')
    ax[0].plot(model_temps_unoise, '.', label='model')
    ax[1].plot(dnoisy.vector-model_temps_unoise, '.')
    ax[0].set_xlabel('bin')
    ax[1].set_xlabel('bin')
    ax[0].set_ylabel('Temperature [K]')
    ax[1].set_ylabel('Temperature residuals [K]')
    ax[0].legend()
    fig.tight_layout()
    fig.savefig('fig/non_uniform_noise_comparison_1.png')
    plt.close('all')

    # Plot the power law indices.
    probable_theta = np.array([0.5*(bound[0]+bound[1]) for bound in inference_bounds])
    plt.plot(theta_mean, 'o', label='inferred radiometric')
    plt.plot(theta_mean_unoise, 'o', label='inferred uniform')
    plt.plot(probable_theta, '.', label='predicted')
    plt.xlabel("parameter")
    plt.ylabel("power law index")
    plt.legend()
    plt.savefig('fig/non_uniform_noise_comparison_3.png')


def plot_inference(fname, burn_in=10000):
    """
    Make plots showcasing the inference.
    """
    from chainconsumer import ChainConsumer
    chain = np.load("saves/Nregs_pl_gsmalo/"+fname+".npy")
    c=ChainConsumer()
    chain = chain[burn_in:]
    ch_sh = np.shape(chain)
    chain_flat = np.reshape(chain, (ch_sh[0]*ch_sh[1], ch_sh[2]))  # Flatten chain.
    c.add_chain(chain_flat, parameters=[r'$\gamma$'+f'$_{i}$' for i in range(1, len(chain_flat[0])+1)])
    c.plotter.plot()
    plt.show()

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
