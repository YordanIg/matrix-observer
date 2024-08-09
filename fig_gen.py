"""
Generate the figures (I think) will end up in the paper.
"""
from functools import partial
import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from chainconsumer import ChainConsumer
import healpy as hp

import src.observing as OBS
import src.sky_models as SM
import src.beam_functions as BF
import src.forward_model as FM
from src.blockmat import BlockMatrix
from src.spherical_harmonics import RealSphericalHarmonics
RS = RealSphericalHarmonics()

from binwise_modelling import fg_cm21_chrom_corr

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

def gen_binwise_achrom():
    # Four-antenna case:
    fg_cm21_chrom_corr(Npoly=9, mcmc=True, chrom=None, savetag="", lats=np.array([-26*2, -26, 26, 26*2]))
    # Single-antenna case:
    fg_cm21_chrom_corr(Npoly=9, mcmc=True, chrom=None, savetag="", lats=np.array([-26]))

def plot_binwise_achrom():
    nant4_mcmcChain = np.load('saves/Binwise/Nant<4>_achrom_mcmcChain.npy')
    nant4_mlChain = np.load('saves/Binwise/Nant<4>_achrom_mlChain.npy')

    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    c.add_chain(nant4_mlChain, parameters=['A', 'nu0', 'dnu'])
    c.add_chain(nant4_mcmcChain[:,-3:])
    f = c.plotter.plot()
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(nant4_mcmcChain))), size=1000)
    samples_mcmcChain = nant4_mcmcChain[idx_mcmcChain]
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
    
    idx_mlChain = np.random.choice(a=list(range(len(nant4_mlChain))), size=1000)
    samples_mlChain = nant4_mlChain[idx_mlChain]
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

def gen_binwise_chrom():
    '''# Four-antenna case:
    fg_cm21_chrom_corr(Npoly=9, mcmc=True, chrom=1.6e-2, savetag="", lats=np.array([-26*2, -26, 26, 26*2]))
    # Single-antenna case:
    fg_cm21_chrom_corr(Npoly=9, mcmc=True, chrom=1.6e-2, savetag="", lats=np.array([-26]))'''

    # Investigate which chromaticities have any chance of working using ML method. Combine this with basemap errors (and look at Npoly)
    startpos = np.append(np.mean(np.load('saves/Binwise/Nant<4>_Npoly<9>_chrom<3.4e-02>_mcmcChain.npy'), axis=0)[:-4], OBS.cm21_params)
    fg_cm21_chrom_corr(Npoly=8, mcmc=True, chrom=3.4e-2, savetag="", lats=np.array([-26*2, -26, 26, 26*2]), mcmc_pos=startpos)

def plot_binwise_chrom():
    nant4_mcmcChain = np.load('saves/Binwise/Nant<4>_Npoly<8>_chrom<3.4e-02>_mcmcChain.npy')
    nant4_mlChain = np.load('saves/Binwise/Nant<4>_Npoly<8>_chrom<3.4e-02>_mlChain.npy')

    # Standard marginalised corner plot of the 21-cm monopole parameters.
    c = ChainConsumer()
    c.add_chain(nant4_mlChain, parameters=['A', 'nu0', 'dnu'])
    c.add_chain(nant4_mcmcChain[:,-3:])
    f = c.plotter.plot()
    plt.show()

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(nant4_mcmcChain))), size=1000)
    samples_mcmcChain = nant4_mcmcChain[idx_mcmcChain]
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
    
    idx_mlChain = np.random.choice(a=list(range(len(nant4_mlChain))), size=1000)
    samples_mlChain = nant4_mlChain[idx_mlChain]
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
