"""
Reconstructing both GSMA and Gaussian random field skies using the model:
    d = PYB a + n
"""
from pygdsm import GlobalSkyModel2016
import healpy as hp
import numpy as np
from numpy.linalg import svd
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
import seaborn as sns

import src.beam_functions as BF
import src.spherical_harmonics as SH
import src.forward_model as FM
import src.sky_models as SM
import src.map_making as MM
import src.plotting as PL

RS = SH.RealSphericalHarmonics()


def calc_full_unmodelled_mode_matrix(lmod, lmax, nside, foreground_power_spec, beam_mat, binning_mat, pointing_mat):
    """
    Calculate the unmodelled mode matrix for the non-trivial observation 
    strategy and binning of timeseries data into Ntau bins.
    """
    npix = hp.nside2npix(nside)
    vectors = hp.pix2vec(nside, ipix=list(range(npix)))
    vectors = np.array(vectors).T
    vector_difference = np.einsum("pi,qi->pq", vectors, vectors)
    binpoint_mat = binning_mat@pointing_mat
    val = np.sum([((2*l+1)/(4*np.pi)) * eval_legendre(l, vector_difference) * foreground_power_spec[l] * beam_mat[l,l]**2 for l in range(lmod+1, lmax)], axis=0)
    return binpoint_mat@val@(binpoint_mat.T)


def reconstruct_obs(noise=True, obs_strat="nontriv", which_fg='gsma', missing_modes=False, extra_tag='', **kwargs):
    """
    Reconstruct the single-frequency alm vector using maximum-likelihood
    estimation for a Gaussian random field with the same power spectrum as the
    GSMA.

    The generation of the alm phases of the Gaussian field is stochastic, but 
    each time the function is run it starts with the same seed, so the results
    won't change between runs.
    """
    default_args = {
        'lmax'     : 32,
        'lmod_arr' : [20, 15, 11, 8],
        'nside'    : 16,
        'nu'       : 60,
        'lat_arr'  : np.linspace(-89, 89, 19)
    }
    fargs = {}

    for arg in default_args:
        if arg in kwargs:
            fargs[arg] = kwargs[arg]
        else:
            fargs[arg] = default_args[arg]

    npix = hp.nside2npix(fargs['nside'])

    # Set up the foreground alm models.
    if which_fg == 'gauss':
        # Generate a Gaussian random field with same power spectrum as GSMA.
        gsma_alm = SM.foreground_gsma_alm(nu=fargs['nu'])
        gsma_complex_alm = RS.real2ComplexALM(gsma_alm)
        fg_cl = hp.alm2cl(gsma_complex_alm)
        fg_complex_alm = hp.synalm(fg_cl, lmax=fargs['lmax'])
        fg_alm = RS.complex2RealALM(fg_complex_alm)
    if which_fg == 'gsma':
        fg_alm = SM.foreground_gsma_alm(nu=fargs['nu'], lmax=fargs['lmax'])
        fg_complex_alm = RS.real2ComplexALM(fg_alm)
        fg_cl = hp.alm2cl(fg_complex_alm)

    # Specify observation strategy and build observation matrix.
    narrow_cosbeam = lambda x : BF.beam_cos(x, theta0=0.8)
    if obs_strat == 'nontriv':
        times = np.linspace(0, 24, 40, endpoint=False)
        lats = fargs['lat_arr']
        mat_A_fm, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan(nside=fargs['nside'], lmax=fargs['lmax'], lats=lats, beam_use=narrow_cosbeam, times=times, return_mat=True)
    elif obs_strat == 'triv':
        mat_A_fm, (mat_G, mat_Y, mat_B) = FM.calc_observation_matrix_all_pix(nside=fargs['nside'], lmax=fargs['lmax'], Ntau=npix, Nt=npix, beam_use=narrow_cosbeam, return_mat=True)
        mat_P = np.diag([1]*npix)
    else:
        raise ValueError("invalid observation strategy choice.")

    # Perform fiducial observations and add noise.
    d = mat_A_fm @ fg_alm
    d_noisy, noise_covar = SM.add_noise(temps=d, dnu=1, Ntau=len(mat_P[:,0]), 
                                        t_int=1e4, seed=124)

    # Set up model inversion matrices.
    mat_A_mod_arr  = []
    mat_W_arr      = []
    cov_arr        = []
    a_estimate_arr = []
    for lmod in fargs['lmod_arr']:
        if obs_strat == 'nontriv':
            mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan(
                nside=fargs['nside'], 
                lmax=lmod, 
                lats=lats, 
                beam_use=narrow_cosbeam, 
                times=times, 
                return_mat=False
            )
        elif obs_strat == 'triv':
            mat_A_mod = FM.calc_observation_matrix_all_pix(nside=fargs['nside'], lmax=lmod, Ntau=npix, Nt=npix, beam_use=narrow_cosbeam)

        if missing_modes:
            mat_S = calc_full_unmodelled_mode_matrix(lmod=lmod, lmax=fargs['lmax'], nside=fargs['nside'], foreground_power_spec=fg_cl, beam_mat=mat_B, binning_mat=mat_G, pointing_mat=mat_P)
            full_noise_covar = noise_covar + mat_S
        else:
            full_noise_covar = noise_covar
        mat_W, cov = MM.calc_ml_estimator_matrix(mat_A=mat_A_mod, 
                                                 mat_N=full_noise_covar, cov=True)

        if noise:
            a_estimate = mat_W @ d_noisy
        else:
            a_estimate = mat_W @ d

        mat_A_mod_arr.append(mat_A_mod)
        mat_W_arr.append(mat_W)
        cov_arr.append(cov)
        a_estimate_arr.append(a_estimate)
    
    # Generate plot to compare reconstructions.
    if noise:
        noisetag = 'noisy'
    else:
        noisetag = 'noisefree'
    if which_fg == 'gauss':
        fgtag = 'gaussFG'
    elif which_fg == 'gsma':
        fgtag = 'GSMA'
    if obs_strat=='triv':
        obstag = 'trivobs'
    elif obs_strat=='nontriv':
        obstag = 'nontrivobs'
    if missing_modes:
        mmmtag = 'corr'
    elif not missing_modes:
        mmmtag = 'uncorr'

    tag  = f"monofreq_reconstruct_{fgtag}_{obstag}_{noisetag}_{mmmtag}_{extra_tag}.pdf"
    labs = [f"lmod={lmod}" for lmod in fargs['lmod_arr']]
    fmts = ['.', 'x', 'o', '-', '.', '.', '.', '.', '.', '.']

    fig = PL.compare_reconstructions(fg_alm, *a_estimate_arr, labels=labs, 
                                     fmts=fmts, return_comparisons_fig=True)
    fig.savefig(fname="fig/"+tag)


def gen_all_reconstructions():
    noise_arr = [True, False]
    obs_strat_arr=['nontriv', 'triv']
    which_fg_arr=['gsma', 'gauss']
    missing_modes_arr=[True, False]

    for noise in noise_arr:
        for obs_strat in obs_strat_arr:
            for which_fg in which_fg_arr:
                for missing_modes in missing_modes_arr:
                    reconstruct_obs(noise=noise, obs_strat=obs_strat, 
                                    which_fg=which_fg, 
                                    missing_modes=missing_modes)
                    