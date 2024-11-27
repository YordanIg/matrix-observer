"""
Generate the figures (I think) will end up in the paper.
"""
from functools import partial
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import numpy as np
from chainconsumer import ChainConsumer
import healpy as hp
from matplotlib import cm, colorbar
from matplotlib.colors import Normalize

import src.observing as OBS
import src.sky_models as SM
import src.coordinates as CO
import src.beam_functions as BF
import src.forward_model as FM
from src.blockmat import BlockMatrix
from src.spherical_harmonics import RealSphericalHarmonics, calc_spherical_harmonic_matrix
RS = RealSphericalHarmonics()

from binwise_modelling import fg_cm21_chrom_corr
from multifrequency_ml_modelling import nontrivial_obs_memopt_missing_modes, fg_cm21_polymod

alm2temp = 1/np.sqrt(4*np.pi)
ant_LUT = {
    1 : np.array([-26]),
    2 : np.array([-26, 26]),
    3 : np.array([-26, 0, 26]),
    4 : np.array([-26*2, -26, 26, 26*2]),
    5 : np.array([-26*2, -26, 0, 26, 26*2]),
    6 : np.array([-26*3, -26*2, -26, 26, 26*2, 26*3]),
    7 : np.array([-26*3, -26*2, -26, 0, 26, 26*2, 26*3])
}

################################################################################
# LMOD and NSIDE investigations.
################################################################################
def gen_lmod_investigation():
    def calc_d_vec(lmod=32, nside=64):
        npix    = hp.nside2npix(nside)
        lats = np.array([-26])
        times = np.linspace(0, 24, 3, endpoint=False)
        nuarr   = np.linspace(50,100,51)
        narrow_cosbeam  = lambda x: BF.beam_cos(x, 0.8)
        
        # Generate foreground alm
        fg_alm_mod = SM.foreground_gsma_alm_nsidelo(nu=nuarr, lmax=lmod, nside=nside, use_mat_Y=True)
        
        # Generate observation matrix for the modelling and for the observations.
        mat_A_mod = FM.calc_observation_matrix_multi_zenith_driftscan(nside, lmod, lats=lats, times=times, beam_use=narrow_cosbeam)
        mat_A_mod = BlockMatrix(mat=mat_A_mod, mode='block', nblock=len(nuarr))

        # Calculate RMS errors of the data vectors.
        dmod = mat_A_mod@fg_alm_mod
        return dmod.vector
    pars = [2, 4, 8, 16, 32, 64]
    d_list = []
    for par in pars:
        d_list.append(calc_d_vec(par))
    np.save('INLR_d_list.npy', d_list)

def plot_lmod_nside_investigation():
    """
    Plot the RMS residuals to observing the sky from the LWA site in 3 time bins
    across a single day, compared across multiple LMOD values.
    This list is generated using NSIDE 64, with LMOD=[2, 4, 8, 16, 32, 64].
    The x axis ranges from LMOD=2->32, where e.g. the LMOD=2 point signifies the
    residuals between the LMOD=4 and LMOD=2 observations.

    Includes a second plot showing that the value of NSIDE=32 is enough to
    capture the behaviour of up to LMOD=64 modes, let alone LMOD=32 modes.
    """
    d_list = np.load('INLR_d_list.npy')
    pars = [2, 4, 8, 16, 32, 64]
    # Plot std error between each l value and the next l value, i.e. the first is RMS(l=2 - l=4).
    xx = list(range(len(d_list)-1))
    yy = [np.std(d_list[i]-d_list[i+1]) for i in range(len(d_list)-1)]

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].loglog(pars[:-1],yy, '.')
    ax[0].set_xticks(ticks=[], labels=[], minor=True)
    ax[0].set_xticks(ticks=pars[:-1], labels=pars[:-1], minor=False)
    ax[0].set_ylabel("RMS residual temperature [K]")
    ax[0].set_xlabel(r"$l_\mathrm{mod}$")

    NSIDEs = [2, 4, 8, 16, 32, 64, 128]
    ELLs   = [32, 64]
    rads_NSIDE = [np.sqrt(4*np.pi / (12*NSIDE**2)) for NSIDE in NSIDEs]
    rads_ELL = [2*np.pi/(2*ELL) for ELL in ELLs]
    ax[1].loglog(NSIDEs, rads_NSIDE, '.')
    sty = [':', '-.']
    for ELL, rads, s in zip(ELLs, rads_ELL, sty):
        ax[1].axhline(y=rads, linestyle=s, label="l="+str(ELL), color='k')
    ax[1].legend()
    ax[1].set_xticks(ticks=[], labels=[], minor=True)
    ax[1].set_xticks(ticks=NSIDEs, labels=NSIDEs, minor=False)
    ax[1].set_xlabel("NSIDE")
    ax[1].set_ylabel("Approx pixel width [rad]")
    fig.tight_layout()
    plt.savefig("fig/lmod_nside_investigation.png")
    plt.savefig("fig/lmod_nside_investigation.pdf")

################################################################################
# Basemap error figure.
################################################################################
def plot_basemap_err(save=False):
    # Generate fiducial basemap and basemaps with 5% and 10% errors.
    fid_alm = SM.foreground_gsma_alm_nsidelo(OBS.nuarr, 32, 32, use_mat_Y=True, delta=None)
    err05_alm = SM.foreground_gsma_alm_nsidelo(OBS.nuarr, 32, 32, use_mat_Y=True, delta=SM.basemap_err_to_delta(5))
    err10_alm = SM.foreground_gsma_alm_nsidelo(OBS.nuarr, 32, 32, use_mat_Y=True, delta=SM.basemap_err_to_delta(10))
    err15_alm = SM.foreground_gsma_alm_nsidelo(OBS.nuarr, 32, 32, use_mat_Y=True, delta=SM.basemap_err_to_delta(15))

    # Construct achromatic observation matrix.
    narrow_cosbeam = lambda x: BF.beam_cos(x, theta0=0.8)
    mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr=OBS.nuarr, nside=32, lmax=32, Ntau=1, lats=[-26], times=np.linspace(0, 24, 3, endpoint=False), beam_use=narrow_cosbeam, return_mat=False)

    # Remove the monopole from the alm.
    fid_alm[0::1089] = 0
    err05_alm[0::1089] = 0
    err10_alm[0::1089] = 0
    err15_alm[0::1089] = 0

    # Observe the monopole-less alms
    fid_temp = mat_A@fid_alm
    err05_temp = mat_A@err05_alm
    err10_temp = mat_A@err10_alm
    err15_temp = mat_A@err15_alm

    # Plot the temperature space residuals.
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].axhline(y=0, linestyle=':', color='k')
    ax[0].plot(OBS.nuarr,fid_temp.vector-err05_temp.vector, label='5%')
    ax[0].plot(OBS.nuarr,fid_temp.vector-err10_temp.vector, label='10%')
    ax[0].plot(OBS.nuarr,fid_temp.vector-err15_temp.vector, label='15%')
    ax[0].legend()
    ax[0].set_ylabel("Temperature [K]")
    ax[0].set_xlabel("Frequency [MHz]")

    # Remove the dipole from the alm too
    fid_alm[1::1089] = 0
    err05_alm[1::1089] = 0
    err10_alm[1::1089] = 0
    err15_alm[1::1089] = 0

    fid_alm[2::1089] = 0
    err05_alm[2::1089] = 0
    err10_alm[2::1089] = 0
    err15_alm[2::1089] = 0

    fid_alm[3::1089] = 0
    err05_alm[3::1089] = 0
    err10_alm[3::1089] = 0
    err15_alm[3::1089] = 0

    # Observe the monopole and dipole-less alms
    fid_temp = mat_A@fid_alm
    err05_temp = mat_A@err05_alm
    err10_temp = mat_A@err10_alm
    err15_temp = mat_A@err15_alm

    # Plot the temperature space residuals.
    ax[1].axhline(y=0, linestyle=':', color='k')
    ax[1].plot(OBS.nuarr,fid_temp.vector-err05_temp.vector)
    ax[1].plot(OBS.nuarr,fid_temp.vector-err10_temp.vector)
    ax[1].plot(OBS.nuarr,fid_temp.vector-err15_temp.vector)
    ax[1].set_xlabel("Frequency [MHz]")
    fig.tight_layout()
    if save:
        fig.savefig("fig/basemap_err_mondip.pdf")
    else:
        plt.show()
    plt.close("all")

################################################################################
# Skytrack maps with modelled and unmodelled foreground modes.
################################################################################
def plot_skytrack_maps():
    # Generate the sky tracks of 7 antennas.
    lats = [-3*26, -2*26, -1*26, 0, 1*26, 2*26, 3*26]
    coords = [CO.obs_zenith_drift_scan(lat, lon=0, times=np.linspace(0,24,1000)) for lat in lats]
    nside=256
    _,pix = CO.calc_pointing_matrix(*coords,nside=nside, pixels=True)
    m = np.zeros(hp.nside2npix(nside))
    m[pix] = 1
    hp.mollview(m)
    thetas, phis = hp.pix2ang(nside, pix)
    # Generate the foreground sky alm at 60 MHz.
    fg_alm = SM.foreground_gsma_alm_nsidelo(nu=60, lmax=32, nside=32, use_mat_Y=True)

    # Generate a beam matrix to observe it with.
    mat_B  = BF.calc_beam_matrix(nside=32, lmax=32)

    # Generate/load the spherical harmonic matrix.
    mat_Y  = calc_spherical_harmonic_matrix(nside=32, lmax=32)

    # Split the spherical harmonic matrix into the low and high multipole sections, dividing at lmod.
    lmod  = 3
    Nlmod = RS.get_size(lmod)
    mat_Y_mod   = mat_Y[:,:Nlmod]
    mat_Y_unmod = mat_Y[:,Nlmod:]

    # Convolve the foregrounds with the beam.
    conv_fg = mat_B@fg_alm

    # Transform with the spherical harmonic matrix.
    mod_sky   = mat_Y_mod @ conv_fg[:Nlmod]
    unmod_sky = mat_Y_unmod @ conv_fg[Nlmod:]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6,6))
    ls = (0, (8, 5))
    plt.axes(ax1)
    hp.mollview(mod_sky, hold=True, cbar=None, title=None)
    hp.projplot(thetas, phis, linestyle=ls, color='r', linewidth=2)

    plt.axes(ax2)
    hp.mollview(unmod_sky, hold=True, cbar=None, title=None)
    hp.projplot(thetas, phis, linestyle=ls, color='r', linewidth=2)

    normalize       = Normalize(vmin=np.min(mod_sky), vmax=np.max(mod_sky))
    scalar_mappable = cm.ScalarMappable(norm=normalize)
    colorbar_axis   = fig.add_axes([.16-0.06, .66-0.11, 0.03, .33])  # Colorbar location.
    cbar1 = colorbar.ColorbarBase(colorbar_axis, norm=normalize, 
                        orientation='vertical', ticklocation='left')
    cbar1.set_label(r'Temperature [K]')

    normalize       = Normalize(vmin=np.min(unmod_sky), vmax=np.max(unmod_sky))
    scalar_mappable = cm.ScalarMappable(norm=normalize)
    colorbar_axis   = fig.add_axes([.16-0.06, 0.13, 0.03, .33])  # Colorbar location.
    cbar2 = colorbar.ColorbarBase(colorbar_axis, norm=normalize, 
                        orientation='vertical', ticklocation='left')
    cbar2.set_label(r'Temperature [K]')
    plt.savefig("fig/skytrack_maps.pdf")
    plt.show()

################################################################################
# FWHM plot.
################################################################################
def plot_fwhm():
    """
    Plot the fwhm function from Tauscher et al. 2020 as a function of frequency
    for a range of values of the chromaticity parameter c.
    """
    nu = np.linspace(50, 100, 100)
    fig, ax = plt.subplots()
    c_values = [8.0e-3, 1.6e-2, 2.4e-2, 3.4e-2]
    for c in c_values:
        ax.plot(nu, np.degrees(BF.fwhm_func_tauscher(nu, c)), label=f"c={c:.1e}")
    
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Beam FWHM [deg]")
    ax.legend()
    
    # Adjust x and y limits
    ax.set_xlim([50, 100])
    y_min = np.min([np.degrees(BF.fwhm_func_tauscher(nu, c)) for c in c_values])
    y_max = np.max([np.degrees(BF.fwhm_func_tauscher(nu, c)) for c in c_values])
    y_margin = (y_max - y_min) * 0.05  # Add a 5% margin to the y-axis limits
    ax.set_ylim([y_min - y_margin, y_max])
    
    fig.tight_layout()
    plt.savefig("fig/fwhm.pdf")
    plt.show()


################################################################################
# Alm polynomial forward model inference functions.
################################################################################
def _recompute_mat_A(pars, lmax):
    # Compute the observation matrix.
    narrow_cosbeam = lambda x: BF.beam_cos(x, 0.8)
    if isinstance(pars['chrom'], bool):
        if not pars['chrom']:
            mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr=pars['nuarr'], nside=pars['nside'], lmax=lmax, Ntau=pars['Ntau'], lats=pars['lats'], times=pars['times'], beam_use=narrow_cosbeam)
        elif pars['chrom']:
            mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(pars['nuarr'], pars['nside'], lmax, pars['Ntau'], pars['lats'], pars['times'], beam_use=BF.beam_cos_FWHM, chromaticity=BF.fwhm_func_tauscher)
    elif pars['chrom'] is None:
        mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_multifreq(nuarr=pars['nuarr'], nside=pars['nside'], lmax=lmax, Ntau=pars['Ntau'], lats=pars['lats'], times=pars['times'], beam_use=narrow_cosbeam)
    else:
        chromfunc = partial(BF.fwhm_func_tauscher, c=pars['chrom'])
        mat_A = FM.calc_observation_matrix_multi_zenith_driftscan_chromatic(pars['nuarr'], pars['nside'], lmax, pars['Ntau'], pars['lats'], pars['times'], beam_use=BF.beam_cos_FWHM, chromaticity=chromfunc)
    return mat_A

def plot_alm_poly_inference():
    """
    Plotting code for MCMC chains generated by the function
    compare_fm_fid_reconstruction_with21cm.
    """
    # Load 4 runs.
    run0 = {
        'lmax' : 32,
        'lmod' : 0,
        'Npoly': 6,
        'burn_in' : 3000,
        'savetag' : 'chrom2_bmerr10'
    }
    run1 = {
        'lmax' : 32,
        'lmod' : 0,
        'Npoly': 6,
        'burn_in' : 3000,
        'savetag' : 'chrom2_bmerr10'
    }
    run2 = {
        'lmax' : 32,
        'lmod' : 0,
        'Npoly': 6,
        'burn_in' : 3000,
        'savetag' : 'chrom1_bmerr10'
    }
    run3 = {
        'lmax' : 32,
        'lmod' : 0,
        'Npoly': 6,
        'burn_in' : 3000,
        'savetag' : 'chrom3_bmerr10'
    }
    runs = [run0, run1, run2, run3]
    chains = []
    datas  = []
    errors = []
    params = []
    for run in runs:
        # Load and pre=process the chain.
        chain = np.load(f"saves/Alm_corrected/lmax{run['lmax']}_lmod{run['lmod']}_Npoly{run['Npoly']}_{run['savetag']}_chain.npy")
        chain = chain[run['burn_in']:]
        nwalkers, nsteps, ndim = np.shape(chain)
        chain_flat = np.reshape(chain, (nwalkers*nsteps, ndim))
        chains.append(chain_flat)
        # Load mock data and errors.
        datas.append(np.load(f"saves/Alm_corrected/lmax{run['lmax']}_lmod{run['lmod']}_Npoly{run['Npoly']}_{run['savetag']}_data.npy"))
        errors.append(np.load(f"saves/Alm_corrected/lmax{run['lmax']}_lmod{run['lmod']}_Npoly{run['Npoly']}_{run['savetag']}_errs.npy"))
        # Load the model parameters.
        with open(f"saves/Alm_corrected/lmax{run['lmax']}_lmod{run['lmod']}_Npoly{run['Npoly']}_{run['savetag']}_pars.pkl", 'rb') as f:
            parset = pickle.load(f)
        params.append(parset)
    
    
    models = []
    # Re-initialize all the models.
    for parset, run in zip(params, runs):
        mat_A = _recompute_mat_A(pars=parset, lmax=run['lmax'])
        # Compute the correction alms.
        a = SM.foreground_gsma_alm_nsidelo(nu=parset['nuarr'], lmax=run['lmax'], nside=parset['nside'], use_mat_Y=True)
        a_sep = np.array(np.split(a, len(parset['nuarr'])))
        Nlmod = RS.get_size(lmax=run['lmod'])
        alms_for_corr  = a_sep.T[Nlmod:]
        mod = FM.genopt_alm_plfid_forward_model_with21cm(parset['nuarr'], observation_mat=mat_A, fid_alm=alms_for_corr, Npoly=run['Npoly'], lmod=run['lmod'], lmax=run['lmax'])
        models.append(mod)
        
    # Set up the figure.
    fig = plt.figure()
    # Add big invisible subplot for common axes labels.
    ax  = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    # Set up the rest of the subplots.
    spec = gridspec.GridSpec(ncols=2, nrows=4,
                            height_ratios=[2, 1,2,1])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax2 = fig.add_subplot(spec[2])
    ax3 = fig.add_subplot(spec[3])
    ax4 = fig.add_subplot(spec[4])
    ax5 = fig.add_subplot(spec[5])
    ax6 = fig.add_subplot(spec[6])
    ax7 = fig.add_subplot(spec[7])

    ax_sig = [ax0,ax1,ax4,ax5]
    ax_res = [ax2,ax3,ax6,ax7]

    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax.set_ylabel("Temperature [K]")
    ax.set_xlabel("Frequency [MHz]")
    
    # Plot the residuals.
    for a, mod, chain, data, err, parset in zip(ax_res, models, chains, datas, errors, params):
        theta_inferred = np.mean(chain, axis=0)
        d_inferred = mod(theta_inferred)
        residuals = data - d_inferred
        a.errorbar(parset['nuarr'], residuals, err, fmt='.')
    
    # Plot the 21-cm signal.
    for a, mod, chain, data, err, parset in zip(ax_sig, models, chains, datas, errors, params):
        chain_sample_indx = np.random.choice(len(chain), 1000)
        chain_samples = chain[chain_sample_indx]
        chain_samples_cm21 = chain_samples[:,-3:]
        cm21_temps = [FM.cm21_globalT(parset['nuarr'], *theta) for theta in chain_samples_cm21]
        cm21_temps_mean = np.mean(cm21_temps, axis=0)
        cm21_temps_std = np.std(cm21_temps, axis=0)

        a.plot(parset['nuarr'], FM.cm21_globalT(nu=parset['nuarr']), label='fiducial', linestyle=':', color='k')
        a.fill_between(
            parset['nuarr'],
            cm21_temps_mean-cm21_temps_std, 
            cm21_temps_mean+cm21_temps_std,
            color='C1',
            alpha=0.8,
            edgecolor='none',
            label="inferred"
        )
        a.fill_between(
            parset['nuarr'],
            cm21_temps_mean-2*cm21_temps_std, 
            cm21_temps_mean+2*cm21_temps_std,
            color='C1',
            alpha=0.4,
            edgecolor='none'
        )
    
    ax0.legend()
    fig.tight_layout()
    fig.show()

################################################################################
# Achromatic binwise and ML functions (LEGACY?)
################################################################################
def gen_binwise_achrom(Npoly=4):
    # Four-antenna case:
    fg_cm21_chrom_corr(Npoly=Npoly, mcmc=True, chrom=None, savetag="", lats=np.array([-26*2, -26, 26, 26*2]))
    # Single-antenna case:
    #fg_cm21_chrom_corr(Npoly=Npoly, mcmc=True, chrom=None, savetag="", lats=np.array([-26]))

def plot_binwise_achrom(Nant=4, Npoly=6):
    runstr = 'Nant<{}>_Npoly<{}>_achrom'.format(Nant, Npoly)
    mcmcChain = np.load('saves/Binwise/'+runstr+'_mcmcChain.npy')
    mlChain = np.load('saves/Binwise/'+runstr+'_mlChain.npy')
    try:
        bic=np.load('saves/Binwise/'+runstr+'_bic.npy')
        print("MCMC BIC =", bic)
    except:
        pass

    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    c.add_chain(mlChain, parameters=['A', 'nu0', 'dnu'])
    c.add_chain(mcmcChain[:,-3:])
    f = c.plotter.plot()
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)

    plt.plot(OBS.nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        OBS.nuarr,
        a00mean_mcmc-a00std_mcmc, 
        a00mean_mcmc+a00std_mcmc,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        OBS.nuarr,
        a00mean_mcmc-2*a00std_mcmc, 
        a00mean_mcmc+2*a00std_mcmc,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("21-cm a00 [K]")
    plt.legend()
    plt.show()
    
    idx_mlChain = np.random.choice(a=list(range(len(mlChain))), size=1000)
    samples_mlChain = mlChain[idx_mlChain]
    a00list_ml   = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mlChain]
    a00mean_ml = np.mean(a00list_ml, axis=0)
    a00std_ml  = np.std(a00list_ml, axis=0)

    plt.plot(OBS.nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        OBS.nuarr,
        a00mean_ml-a00std_ml, 
        a00mean_ml+a00std_ml,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        OBS.nuarr,
        a00mean_ml-2*a00std_ml, 
        a00mean_ml+2*a00std_ml,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("21-cm a00 [K]")
    plt.legend()
    plt.show()

    chi_sq = np.sum((a00mean_mcmc - cm21_a00)**2 / a00std_mcmc)
    print("monopole chi-sq", chi_sq)

def plot_ml_achrom(Nant=4, Npoly=6, bmerr_idx='0.0851'):
    runstr = 'Nant<{}>_Npoly<{}>_achrom_idx<{}>'.format(Nant, Npoly, bmerr_idx)
    mcmcChain = np.load('saves/MLmod/'+runstr+'_mcmcChain.npy')
    #mlChain = np.load('saves/MLmod/Nant<4>_Npoly<5>_chrom<3.4e-02>_mlChain.npy')
    try:
        bic=np.load('saves/MLmod/'+runstr+'_bic.npy')
        print("MCMC BIC =", bic)
    except:
        pass

    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    #c.add_chain(mlChain, parameters=['A', 'nu0', 'dnu'])
    c.add_chain(mcmcChain[:,-3:])
    f = c.plotter.plot()
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)

    plt.plot(OBS.nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
    plt.fill_between(
        OBS.nuarr,
        a00mean_mcmc-a00std_mcmc, 
        a00mean_mcmc+a00std_mcmc,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    plt.fill_between(
        OBS.nuarr,
        a00mean_mcmc-2*a00std_mcmc, 
        a00mean_mcmc+2*a00std_mcmc,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("21-cm a00 [K]")
    plt.legend()
    plt.show()
    
    chi_sq = np.sum((a00mean_mcmc - cm21_a00)**2 / a00std_mcmc)
    print("monopole chi-sq", chi_sq)

################################################################################
# Chromatic and achromatic binwise modelling functions, with result plotting
# and chi-squared and BIC trend plotting.
################################################################################
def gen_binwise_chrom(Nant=4, Npoly=4, chrom=None, basemap_err=None, savetag=None):
    #startpos = np.append(np.mean(np.load('saves/Binwise/Nant<4>_Npoly<8>_chrom<3.4e-02>_mcmcChain.npy'), axis=0)[:8], OBS.cm21_params)
    chain = np.load('saves/Binwise/Nant<4>_Npoly<10>_chrom<3.4e-02>_mcmcChain.npy')
    c = ChainConsumer().add_chain(chain)
    pars = [elt[1] for elt in c.analysis.get_summary().values()]
    pars = pars[:10]
    startpos = None#np.array(pars)
    fg_cm21_chrom_corr(Npoly=Npoly, mcmc=True, chrom=chrom, savetag=savetag, lats=ant_LUT[Nant], mcmc_pos=startpos, basemap_err=basemap_err, steps=10000, burn_in=5000, fidmap_HS=False)

def run_set_gen_binwise_chrom0_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=None, savetag=savetag)

def plot_set_binwise_chrom0_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=None, savetag=savetag)

def run_set_gen_binwise_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=None, savetag='')

def plot_set_binwise_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=None)

def run_set_gen_binwise_chromsmall_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=None, savetag=savetag)

def plot_set_binwise_chromsmall_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=None, savetag=savetag)

def plot_binwise_chi_sq_bic_chromsmall_bm0(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=None, savetag=savetag)

def run_set_gen_binwise_chrommed_bm0(*Npolys):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=2.4e-2, basemap_err=None, savetag='')

def plot_set_binwise_chrommed_bm0(*Npolys):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='2.4e-02', basemap_err=None)

def plot_binwise_chi_sq_bic_chrommed_bm0(savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=[9, 10, 11], chromstr='2.4e-02', basemap_err=None, savetag=savetag)

def run_set_gen_binwise_chrom_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=None, savetag=savetag)

def plot_set_binwise_chrom_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=4, Npoly=Npoly, chromstr='3.4e-02', basemap_err=None, savetag=savetag)

def plot_binwise_chi_sq_bic_chrom_bm0(savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], chromstr='3.4e-02', basemap_err=None, savetag=savetag)

def run_set_gen_binwise_chrom_bm0_onepoint(*Npolys):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=None, savetag='onepoint')

def plot_set_binwise_chrom_bm0_onepoint(*Npolys):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=None, savetag='onepoint')

def plot_binwise_chi_sq_bic_chrom_bm0_onepoint():
    plot_binwise_chi_sq_bic(Nant=7, Npolys=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], chromstr='3.4e-02', basemap_err=None, savetag='onepoint')

def plot_binwise_chrom(Nant=7, Npoly=7, chromstr='3.4e-02', basemap_err=None, ml_plots=False, savetag=None):
    if ml_plots:
        print("Warning: ML posteriors are not being generated for new runs - this may fail/produce unexpected results.")
    runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
    if chromstr is not None:
        runstr += f"_chrom<{chromstr}>"
    else:
        runstr += f"_achrom"
    if basemap_err is not None:
        runstr += f"_bm<{basemap_err}>"
    if savetag is not None:
        runstr += savetag
    print("loading from", runstr)
    mcmcChain = np.load('saves/Binwise/'+runstr+'_mcmcChain.npy')
    if ml_plots:
        mlChain   = np.load('saves/Binwise/'+runstr+'_mlChain.npy')
    data      = np.load('saves/Binwise/'+runstr+'_data.npy')
    dataerr   = np.load('saves/Binwise/'+runstr+'_dataerr.npy')

    try:
        bic=np.load('saves/Binwise/'+runstr+'_bic.npy')
        print("MCMC BIC =", bic)
    except:
        pass

    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    c.add_chain(mcmcChain[:,-3:], parameters=['A', 'nu0', 'dnu'])
    if ml_plots:
        c.add_chain(mlChain)
    f = c.plotter.plot()
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)
    if ml_plots:        
        idx_mlChain = np.random.choice(a=list(range(len(mlChain))), size=10000)
        samples_mlChain = mlChain[idx_mlChain]
        a00list_ml   = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mlChain]
        a00mean_ml = np.mean(a00list_ml, axis=0)
        a00std_ml  = np.std(a00list_ml, axis=0)

        plt.plot(OBS.nuarr, cm21_a00, label='fiducial', linestyle=':', color='k')
        plt.fill_between(
            OBS.nuarr,
            a00mean_ml-a00std_ml, 
            a00mean_ml+a00std_ml,
            color='C1',
            alpha=0.8,
            edgecolor='none',
            label="inferred"
        )
        plt.fill_between(
            OBS.nuarr,
            a00mean_ml-2*a00std_ml, 
            a00mean_ml+2*a00std_ml,
            color='C1',
            alpha=0.4,
            edgecolor='none'
        )
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("21-cm a00 [K]")
        plt.legend()
        plt.show()

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(4,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax[0].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[1].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel("21-cm Temperature [K]")
    ax[0].legend()

    mat_A_dummy = FM.generate_dummy_mat_A(OBS.nuarr, Ntau=1, lmod=32)
    mod = FM.generate_binwise_cm21_forward_model(nuarr=OBS.nuarr, observation_mat=mat_A_dummy, Npoly=Npoly)
    ax[1].axhline(y=0, linestyle=':', color='k')
    ax[1].errorbar(OBS.nuarr, mod(np.mean(mcmcChain, axis=0))-data, dataerr, fmt='.', color='k')
    ax[1].set_ylabel(r"$T_\mathrm{res}$ [K]")
    fig.tight_layout()
    if savetag is not None:
        plt.savefig(f"fig/Binwise/bw_"+runstr+savetag+".pdf")
        plt.savefig(f"fig/Binwise/bw_"+runstr+savetag+".png")

    plt.show()

    chi_sq = np.sum((a00mean_mcmc - cm21_a00)**2 / a00std_mcmc**2)
    print("monopole chi-sq", chi_sq)
    np.save('saves/Binwise/'+runstr+'_chi_sq.npy', chi_sq)

def plot_binwise_chi_sq_bic(Nant=7, Npolys=[], chromstr='3.4e-02', basemap_err=None, savetag=None):
    runstrs = []
    for Npoly in Npolys:
        runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
        if chromstr is not None:
            runstr += f"_chrom<{chromstr}>"
        else:
            runstr += f"_achrom"
        if basemap_err is not None:
            runstr += f"_bm<{basemap_err}>"
        runstrs.append(runstr)
    chi_sqs = []
    bics    = []
    for runstr in runstrs:
        print("loading from", runstr)
        chi_sqs.append(np.load(f'saves/Binwise/'+runstr+'_chi_sq.npy'))
        bics.append(np.load(f'saves/Binwise/'+runstr+'_bic.npy'))

    fig, ax1 = plt.subplots()
    ax1.axhline(y=1, linestyle=':', color='k')
    ax1.semilogy(Npolys,chi_sqs, color='C0', linestyle='-', marker='o')
    ax1.set_xticks(ticks=Npolys, labels=Npolys)
    ax1.set_xticks(ticks=[], minor=True)
    ax1.set_ylabel(r"21-cm Monpole $\chi^2$")
    ax1.set_xlabel("$N_\mathrm{poly}$")
    ax1.set_xlim([Npolys[0], Npolys[-1]])
    ax2 = ax1.twinx()
    ax2.semilogy(Npolys,bics, color='C1', linestyle='-', marker='s')
    ax2.set_ylabel("Model BIC")
    custom_lines = [
        Line2D([0], [0], color='C0', linestyle='-', marker='o'),
        Line2D([0], [0], color='C1', linestyle='-', marker='s')
    ]
    # Add the custom legend to the plot
    plt.legend(custom_lines, [r'$\chi^2$', 'BIC'])
    fig.tight_layout()
    if savetag is not None:
        s = f"bw_chi_sq_bic_Nant<{Nant}>"
        if chromstr is not None:
            s += f"_chrom<{chromstr}>"
        else:
            s += "_achrom"
        if basemap_err is not None:
            s += f"_bm<{basemap_err}>"
        plt.savefig(f"fig/Binwise/"+s+savetag+".pdf")
        plt.savefig(f"fig/Binwise/"+s+savetag+".png")
    plt.show()

################################################################################
# Chromatic and achromatic ML modelling functions, with result plotting
# and chi-squared and BIC trend plotting.
################################################################################
def gen_ml_chrom(Nant=4, Npoly=6, chrom=None, basemap_err=None):
    startpos = None#np.append(np.mean(np.load('saves/MLmod/Nant<4>_Npoly<8>_chrom<6.0e-02>_idx<0.0851>_mcmcChain.npy'), axis=0)[:Npoly], OBS.cm21_params)
    nontrivial_obs_memopt_missing_modes(Npoly=Npoly, lats=ant_LUT[Nant], chrom=chrom, basemap_err=basemap_err, err_type='idx', mcmc=True, mcmc_pos=startpos, steps=40000, burn_in=13000, plotml=False)

# Achromatic, no basemap error.
def run_set_gen_ml_chrom0_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=0)

def plot_set_ml_chrom0_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=0, savetag=savetag)

def plot_ml_chi_sq_bic_chrom0_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=0, savetag=savetag)

# c=0 chromaticity, no basemap error.
def run_set_gen_ml_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=0)

def plot_set_ml_chromflat_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e-02', basemap_err=0, savetag=savetag)
    
def plot_ml_chi_sq_bic_chromflat_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='0.0e-02', basemap_err=0, savetag=savetag)

# 1.6e-2 chromaticity, no basemap error.
def run_set_gen_ml_chromsmall_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=0)

def plot_set_ml_chromsmall_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=0, savetag=savetag)
    
def plot_ml_chi_sq_bic_chromsmall_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=0, savetag=savetag)

# 3.4e-2 chromaticity, no basemap error.
def run_set_gen_ml_chrom_bm0(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=0)

def plot_set_ml_chrom_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=0, savetag=savetag)
    
def plot_ml_chi_sq_bic_chrom_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=0, savetag=savetag)

# Achromatic, 5% basemap error.
def run_set_gen_ml_chrom0_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=5)

def plot_set_ml_chrom0_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chrom0_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=5, savetag=savetag)

# c=0 chromaticity, 5% basemap error.
def run_set_gen_ml_chromflat_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=5)

def plot_set_ml_chromflat_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e-02', basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chromflat_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='0.0e-02', basemap_err=5, savetag=savetag)

# 1.6e-2 chromaticity, 5% basemap error.
def run_set_gen_ml_chromsmall_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=5)

def plot_set_ml_chromsmall_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chromsmall_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

# 3.4e-2 chromaticity, 5% basemap error.
def run_set_gen_ml_chrom_bm5(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=5)

def plot_set_ml_chrom_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=5, savetag=savetag)

def plot_ml_chi_sq_bic_chrom_bm5(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=5, savetag=savetag)


def plot_ml_chrom(Nant=7, Npoly=7, chromstr=None, basemap_err=None, savetag=None):
    runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
    if chromstr is not None:
        runstr += f"_chrom<{chromstr}>"
    else:
        runstr += f"_achrom"
    if basemap_err is not None:
        runstr += f"_idx<{basemap_err}>"
    mcmcChain = np.load('saves/MLmod/'+runstr+'_mcmcChain.npy')
    residuals = np.load('saves/MLmod/'+runstr+'_modres.npy')
    data      = np.load('saves/MLmod/'+runstr+'_data.npy')
    dataerr   = np.load('saves/MLmod/'+runstr+'_dataerr.npy')
    fid_a00   = np.load("saves/MLmod/"+runstr+"_fid_a00.npy")
    rec_a00   = np.load("saves/MLmod/"+runstr+"_rec_a00.npy")
    a00_error = np.load("saves/MLmod/"+runstr+"_rec_a00_err.npy")

    try:
        bic=np.load('saves/MLmod/'+runstr+'_bic.npy')
        print("MCMC BIC =", bic)
    except:
        pass

    # Calculate number of timeseries data points per antenna to reshape the data
    # arrays.
    Nfreq = len(OBS.nuarr)
    Ntau  = int(len(data) / (Nfreq*Nant))
    data  = np.reshape(data, (Nfreq, Nant, Ntau))
    dataerr   = np.reshape(dataerr, (Nfreq, Nant, Ntau))
    residuals = np.reshape(residuals, (Nfreq, Nant, Ntau))
    
    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    c.add_chain(mcmcChain[:,-3:], parameters=[r'$A_{21}$', r'$nu_{21}$', r'$\Delta$'])
    f = c.plotter.plot(truth=[*OBS.cm21_params])
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(4,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax[0].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[0].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[1].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel(r"$T_\mathrm{mon}^{21}$ [K]")
    ax[0].legend()

    ax[1].axhline(y=0, linestyle=':', color='k')
    ax[1].errorbar(OBS.nuarr, (rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp, a00_error*alm2temp, fmt='.', color='k')
    ax[1].axhline(0, linestyle=':', color='k')
    ax[1].set_ylabel(r"$\hat{T}_\mathrm{mon}-\mathcal{M}$ [K]")
    fig.tight_layout()
    if savetag is not None:
        plt.savefig(f"fig/MLmod/ml_"+runstr+savetag+".pdf")
        plt.savefig(f"fig/MLmod/ml_"+runstr+savetag+".png")
    plt.show()

    plt.errorbar(OBS.nuarr, (fid_a00-rec_a00)*alm2temp, a00_error*alm2temp, fmt='.')
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"$T_\mathrm{mon} - \hat{T}_\mathrm{mon}$ [K]")
    if savetag is not None:
        plt.savefig(f"fig/MLmod/ml_"+runstr+savetag+"_inferred_Tmon_res.pdf")
        plt.savefig(f"fig/MLmod/ml_"+runstr+savetag+"_inferred_Tmon_res.png")
    plt.show()
    
    chi_sq = np.sum((a00mean_mcmc - cm21_a00)**2 / a00std_mcmc**2)
    print("monopole chi-sq", chi_sq)
    np.save('saves/MLmod/'+runstr+'_chi_sq.npy', chi_sq)


def plot_ml_chi_sq_bic(Nant=4, Npolys=[], chromstr='3.4e-02', basemap_err=None, savetag=None):
    runstrs = []
    for Npoly in Npolys:
        runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
        if chromstr is not None:
            runstr += f"_chrom<{chromstr}>"
        else:
            runstr += f"_achrom"
        if basemap_err is not None:
            runstr += f"_idx<{basemap_err}>"
        runstrs.append(runstr)
    chi_sqs = []
    bics    = []
    for runstr in runstrs:
        chi_sqs.append(np.load(f'saves/MLmod/'+runstr+'_chi_sq.npy'))
        bics.append(np.load(f'saves/MLmod/'+runstr+'_bic.npy'))

    fig, ax1 = plt.subplots()
    ax1.axhline(y=1, linestyle=':', color='k')
    ax1.semilogy(Npolys,chi_sqs, color='C0', linestyle='-', marker='o')
    ax1.set_xticks(ticks=Npolys, labels=Npolys)
    ax1.set_xticks(ticks=[], minor=True)
    ax1.set_ylabel(r"21-cm Monpole $\chi^2$")
    ax1.set_xlabel("$N_\mathrm{poly}$")
    ax1.set_xlim([Npolys[0], Npolys[-1]])
    ax2 = ax1.twinx()
    ax2.semilogy(Npolys,bics, color='C1', linestyle='-', marker='s')
    ax2.set_ylabel("Model BIC")
    custom_lines = [
        Line2D([0], [0], color='C0', linestyle='-', marker='o'),
        Line2D([0], [0], color='C1', linestyle='-', marker='s')
    ]
    # Add the custom legend to the plot
    plt.legend(custom_lines, [r'$\chi^2$', 'BIC'])
    fig.tight_layout()
    if savetag is not None:
        s = f"ml_chi_sq_bic_Nant<{Nant}>"
        if chromstr is not None:
            s += f"_chrom<{chromstr}>"
        else:
            s += "_achrom"
        if basemap_err is not None:
            s += f"_idx<{basemap_err}>"
        plt.savefig(f"fig/MLmod/"+s+savetag+".pdf")
        plt.savefig(f"fig/MLmod/"+s+savetag+".png")
    plt.show()

def run_all_ml():
    """
    Batch run all run_set_gen_ml functions in the script for Npoly=3, ..., 7.
    """
    run_set_gen_ml_chrom0_bm0(3, 4, 5, 6, 7)
    run_set_gen_ml_chromflat_bm0(3, 4, 5, 6, 7)
    run_set_gen_ml_chromsmall_bm0(3, 4, 5, 6, 7)
    run_set_gen_ml_chrom_bm0(3, 4, 5, 6, 7)
    run_set_gen_ml_chrom0_bm5(3, 4, 5, 6, 7)
    run_set_gen_ml_chromflat_bm5(3, 4, 5, 6, 7)
    run_set_gen_ml_chromsmall_bm5(3, 4, 5, 6, 7)
    run_set_gen_ml_chrom_bm5(3, 4, 5, 6, 7)
