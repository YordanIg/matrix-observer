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
import src.map_making as MM
from src.blockmat import BlockMatrix
from src.spherical_harmonics import RealSphericalHarmonics, calc_spherical_harmonic_matrix
from src.plotting import AxesCornerPlot
from anstey.generate import T_CMB
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
        lats  = ant_LUT[7]
        times = np.linspace(0, 24, 12, endpoint=False)
        nuarr   = np.array([70])
        narrow_cosbeam  = lambda x: BF.beam_cos_FWHM(x, FWHM=np.radians(60))
        
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

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
    ax[0].loglog(pars[:-1],yy)
    ax[0].set_xticks(ticks=[], labels=[], minor=True)
    ax[0].set_xticks(ticks=pars[:-1], labels=pars[:-1], minor=False)
    ax[0].axhline(y=0.1, linestyle=':', color='k')
    ax[0].text(x=2.5,y=0.1*1.1, s="21-cm signal scale")
    ax[0].set_xlim(pars[0], pars[-2])
    ax[0].set_ylim(yy[-1], yy[0])
    ax[0].set_ylabel("RMS residual temperature [K]")
    ax[0].set_xlabel(r"$l_\mathrm{max}$")

    NSIDEs = [2, 4, 8, 16, 32, 64, 128]
    ELLs   = [32, 64]
    rads_NSIDE = [np.sqrt(4*np.pi / (12*NSIDE**2)) for NSIDE in NSIDEs]
    rads_ELL = [2*np.pi/(2*ELL) for ELL in ELLs]
    ax[1].loglog(NSIDEs, np.degrees(rads_NSIDE))
    sty = ['--', '-.']
    for ELL, rads, s in zip(ELLs, rads_ELL, sty):
        ax[1].axhline(y=np.degrees(rads), linestyle=s, color='k')
        ax[1].text(x=64, y=np.degrees(rads)*1.05, s="$l=$"+str(ELL), horizontalalignment='center')
    ax[1].set_xticks(ticks=[], labels=[], minor=True)
    ax[1].set_xticks(ticks=NSIDEs, labels=NSIDEs, minor=False)
    ax[1].set_xlim(NSIDEs[0], NSIDEs[-1])
    ax[1].set_ylim(np.degrees(rads_NSIDE[-1]), np.degrees(rads_NSIDE[0]))
    ax[1].set_xlabel("NSIDE")
    ax[1].set_ylabel("Approx pixel width [deg]")
    fig.tight_layout()
    plt.savefig("fig/lmod_nside_investigation.png")
    plt.savefig("fig/lmod_nside_investigation.pdf")
    plt.show()

def plot_lmod_investigation():
    """
    Plot the RMS residuals to observing the sky from the LWA site in 3 time bins
    across a single day, compared across multiple LMOD values.
    This list is generated using NSIDE 64, with LMOD=[2, 4, 8, 16, 32, 64].
    The x axis ranges from LMOD=2->32, where e.g. the LMOD=2 point signifies the
    residuals between the LMOD=4 and LMOD=2 observations.

    Does not include the second plot that plot_lmod_nside_investigation does.
    """
    d_list = np.load('INLR_d_list.npy')
    pars = [2, 4, 8, 16, 32, 64]
    # Plot std error between each l value and the next l value, i.e. the first is RMS(l=2 - l=4).
    xx = list(range(len(d_list)-1))
    yy = [np.std(d_list[i]-d_list[i+1]) for i in range(len(d_list)-1)]

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.loglog(pars[:-1],yy)
    ax.set_xticks(ticks=[], labels=[], minor=True)
    ax.set_xticks(ticks=pars[:-1], labels=pars[:-1], minor=False)
    ax.axhline(y=0.1, linestyle=':', color='k')
    ax.text(x=2.5,y=0.1*1.1, s="21-cm signal scale")
    ax.set_xlim(pars[0], pars[-2])
    ax.set_ylim(yy[-1], yy[0])
    ax.set_ylabel("RMS residual temperature [K]")
    ax.set_xlabel(r"$l_\mathrm{max}$")
    fig.tight_layout()
    fig.savefig("fig/lmod_investigation.png")
    fig.savefig("fig/lmod_investigation.pdf")
    plt.show()

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
    lmod  = 5
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
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    c_values = [0, 1.6e-2, 3.4e-2, 5.2e-2]
    lss = [':', '--', '-.', (0, (6.4, 1.6, 1.0, 1.6, 1.0, 1.6))]
    cols = ["k", "C0", "C1", "C2"]
    for c, ls, col in zip(c_values, lss, cols):
        ax.plot(nu, np.degrees(BF.fwhm_func_tauscher(nu, c)), linestyle=ls, color=col)
    ax.axhline(y=72, linestyle='-', color='k')
    ax.text(x=60, y=72.2, s="achromatic")
    ax.text(x=62.9, y=65.8, s="0.0e-02", rotation=33)
    ax.text(x=65.8, y=62.5, s="1.6e-02", rotation=28)
    ax.text(x=69.5, y=58.1, s="3.4e-02", rotation=30)
    ax.text(x=73.6, y=54, s="5.2e-02", rotation=38)
    
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Beam FWHM [deg]")
    
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
# 
################################################################################
def plot_basemap_errs():
    def simp_basemap_err_to_delta(bmerr, ref_freq=70):
        return (bmerr/100)/np.log(408/ref_freq)
    def gaussian(x, sig, Nside=32):
        Npix = hp.nside2npix(Nside)
        return 2.35*Npix/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*x**2/sig**2)

    nuarr = OBS.nuarr
    delta_10 = simp_basemap_err_to_delta(10, ref_freq=70)
    delta_20 = simp_basemap_err_to_delta(20, ref_freq=70)

    delta_to_err = lambda delta: np.log(408/nuarr)*delta
    percentage_err_10 = delta_to_err(delta_10)
    percentage_err_20 = delta_to_err(delta_20)

    select_freqs = np.array([60,75,90])
    gsma  = SM.foreground_gsma_nsidelo(nu=select_freqs, nside=32)
    _, err_10 = SM.foreground_gsma_alm_nsidelo(nu=select_freqs, lmax=32, nside=32, original_map=True, delta=delta_10)
    _, err_20 = SM.foreground_gsma_alm_nsidelo(nu=select_freqs, lmax=32, nside=32, original_map=True, delta=delta_20)

    err_mean_10 = []
    for nu, gsma_map in zip(select_freqs, gsma):
        sigma_T   = delta_10 * np.log(408/nu)
        temp_mean_block = (gsma_map - T_CMB) * np.exp(sigma_T**2/2) + T_CMB
        err_mean_10.append(temp_mean_block)
    
    err_mean_20 = []
    for nu, gsma_map in zip(select_freqs, gsma):
        sigma_T   = delta_20 * np.log(408/nu)
        temp_mean_block = (gsma_map - T_CMB) * np.exp(sigma_T**2/2) + T_CMB
        err_mean_20.append(temp_mean_block)

    fig, ax = plt.subplots(1, 2, figsize=(6,2.8))
    ax[0].plot(nuarr, 1e2*percentage_err_10, label='10%')
    ax[0].plot(nuarr, 1e2*percentage_err_20, label='20%')
    ax[0].set_xlabel("Frequency [MHz]")
    ax[0].set_ylabel("Fractional Std Dev [%]")
    ax[0].set_xlim(nuarr[0], nuarr[-1])
    ax[0].set_ylim(0, 1e2*np.max(percentage_err_20))

    bins = np.linspace(-50,50,41)
    ax[1].hist(1e2*(err_mean_20[0]-err_20[0])/err_mean_20[0], bins=bins, ec='k', alpha=0.5, color='C1', label=' 20%')
    ax[1].plot(bins, gaussian(bins, 20), color='C1',linewidth=1.5)
    ax[1].hist(1e2*(err_mean_10[0]-err_10[0])/err_mean_10[0], bins=bins, ec='k', alpha=0.5, color='C0', label=' 10%')
    ax[1].plot(bins, gaussian(bins, 10), color='C0',linewidth=1.5)
    ax[1].set_xlim(-50,50)
    ax[1].set_xlabel("Fractional Pixel Deviation [%]")
    ax[1].set_ylabel("Count")
    ax[1].legend()
    fig.tight_layout()
    plt.savefig("fig/basemap_errs.pdf")
    plt.savefig("fig/basemap_errs.png")
    plt.show()

################################################################################
# Monopole reconstruction error figure.
################################################################################
def plot_monopole_reconstruction_err():
    """
    Showcase the monopole reconstruction error when we reconstruct only modes
    up to lmod for GSMA truncated at lmod, and when we reconstruct only modes
    up to lmod for a non-truncated GSMA.
    """
    # Generate single-frequency noisy foregrounds.
    fg = SM.foreground_gsma_alm_nsidelo(nu=70, lmax=32, nside=32, use_mat_Y=True)

    # Truncate this at various ell values.
    ell_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    N_arr   = [RS.get_size(ell) for ell in ell_arr]
    fg_truncs = [fg[:N] for N in N_arr]

    def calc_mon_err(lats):
        # Generate observation matrix for a number of antennas.
        times = np.linspace(0, 24, 24, endpoint=False)
        narrow_cosbeam = lambda x: BF.beam_cos(x, 0.8)
        mat_A, (mat_G, mat_P, mat_Y, mat_B) = FM.calc_observation_matrix_multi_zenith_driftscan(nside=32, lmax=32, 
                                                        lats=lats, 
                                                        times=times, beam_use=narrow_cosbeam, return_mat=True)

        # Observe for various ell values
        mat_A_truncs = [mat_A[:,:N] for N in N_arr]#[mat_P@mat_Y_i@mat_B_i for mat_Y_i,mat_B_i in zip(mat_Y_truncs, mat_B_truncs)]
        d_truncs = [mat_A_i@fg_i for mat_A_i,fg_i in zip(mat_A_truncs,fg_truncs)]

        # Add noise.
        d_noise_andcov_truncs = [SM.add_noise(d, dnu=1, Ntau=len(times), t_int=200, seed=456) for d in d_truncs]
        d_noise_truncs, d_cov_truncs = map(list, zip(*d_noise_andcov_truncs))
        print(f"noise mag for Nant {len(lats)} is {np.sqrt(np.mean([np.mean(d_cov) for d_cov in d_cov_truncs]))}")
        # Compute the maxlike estimator matrix for each case.
        mat_W_truncs = [MM.calc_ml_estimator_matrix(mat_A_i, mat_N_i, cond=True) for mat_A_i, mat_N_i in zip(mat_A_truncs, d_cov_truncs)]

        # Reconstruct the alm for each truncation case.
        alm_rec_truncs = [mat_W_i @ d_noise_i for mat_W_i,d_noise_i in zip(mat_W_truncs,d_noise_truncs)]
        alm_rec_truncs_nonoise = [mat_W_i @ d_i for mat_W_i,d_i in zip(mat_W_truncs,d_truncs)]

        # Visualise the reconstruction error for the monopole in each case.
        mon_err = [np.abs((alm_rec_i[0]-fg_i[0])*alm2temp) for fg_i,alm_rec_i in zip(fg_truncs,alm_rec_truncs)]
        mon_err_nonoise = [np.abs((alm_rec_i[0]-fg_i[0])*alm2temp) for fg_i,alm_rec_i in zip(fg_truncs,alm_rec_truncs_nonoise)]

        # Visualise the reconstruction error for the lmod=5 case.
        alm_err_lmod5 = np.abs((alm_rec_truncs[5]-fg_truncs[5])*alm2temp)

        return mon_err, mon_err_nonoise, alm_err_lmod5
    
    mon_err7, mon_err_nonoise7, alm_err_lmod5_7 = calc_mon_err([-3*26, -2*26, -1*26, 0, 1*26, 2*26, 3*26])
    mon_err5, mon_err_nonoise5, alm_err_lmod5_5 = calc_mon_err([-2*26, -1*26, 0, 1*26, 2*26])
    mon_err3, mon_err_nonoise3, alm_err_lmod5_3 = calc_mon_err([-1*26, 0, 26])

    
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
    lss = [':', '--', '-.']
    ax[0].semilogy(ell_arr,mon_err_nonoise7, linestyle=lss[0], color='k', alpha=0.4)
    ax[0].plot(ell_arr,mon_err_nonoise5, linestyle=lss[1], color='k', alpha=0.4)
    ax[0].plot(ell_arr,mon_err_nonoise3, linestyle=lss[2], color='k', alpha=0.4)
    ax[0].plot(ell_arr,mon_err7, label=r'$N_\mathrm{ant}$=7', linestyle=lss[0])
    ax[0].plot(ell_arr,mon_err5, label=r'$N_\mathrm{ant}$=5', linestyle=lss[1])
    ax[0].plot(ell_arr,mon_err3, label=r'$N_\mathrm{ant}$=3', linestyle=lss[2])
    ax[0].axhline(y=0.1, color='k')
    ax0_majorticks = list(range(0,ell_arr[-1]+1,2))
    ax[0].set_xticks(ticks=ax0_majorticks, minor=False)
    ax[0].set_xticks(ticks=ell_arr, minor=True)
    ax[0].set_xlabel(r"$l_\mathrm{mod}$")
    ax[0].set_ylabel(r"Monopole Reconstruction Error [K]")
    ax[0].set_xlim(0,12)
    ax[0].set_ylim(0.004*1e-3, 1000*1e-3)#(0.4, 1000)
    ax[0].legend()

    ax[1].axhline(y=0.1, color='k')
    ax[1].semilogy(alm_err_lmod5_7, linestyle=lss[0])
    ax[1].semilogy(alm_err_lmod5_5, linestyle=lss[1])
    ax[1].semilogy(alm_err_lmod5_3, linestyle=lss[2])
    ax1_minorticks = list(range(0,len(alm_err_lmod5_7)))
    ax[1].set_xticks(ticks=ax1_minorticks, minor=True)
    ax[1].set_xlim(0,len(alm_err_lmod5_7)-1)
    ax[1].set_ylabel(r"Multipole Reconstruction Error [K]")
    ax[1].set_xlabel(r"$\mathbf{a}$ vector index")

    fig.tight_layout()
    plt.savefig("fig/monopole_reconstruction_err.pdf")
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
def construct_runstr(Nant, Npoly, chromstr, basemap_err):
    runstr    = f"Nant<{Nant}>_Npoly<{Npoly}>"
    if chromstr is not None:
        runstr += f"_chrom<{chromstr}>"
    else:
        runstr += f"_achrom"
    if basemap_err is not None:
        runstr += f"_idx<{basemap_err}>"
    return runstr

def gen_binwise_chrom(Nant=4, Npoly=4, chrom=None, basemap_err=None, savetag=None):
    #startpos = np.append(np.mean(np.load('saves/Binwise/Nant<4>_Npoly<8>_chrom<3.4e-02>_mcmcChain.npy'), axis=0)[:8], OBS.cm21_params)
    chain = np.load('saves/Binwise/Nant<4>_Npoly<10>_chrom<3.4e-02>_mcmcChain.npy')
    c = ChainConsumer().add_chain(chain)
    pars = [elt[1] for elt in c.analysis.get_summary().values()]
    pars = pars[:10]
    startpos = None#np.array(pars)
    fg_cm21_chrom_corr(Npoly=Npoly, mcmc=True, chrom=chrom, savetag=savetag, lats=ant_LUT[Nant], mcmc_pos=startpos, basemap_err=basemap_err, steps=200000, burn_in=175000, fidmap_HS=False)

# Achromatic, no basemap error.
def run_set_gen_binwise_chrom0_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=None, savetag=savetag)

def plot_set_binwise_chrom0_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=None, savetag=savetag)

def plot_binwise_chi_sq_bic_chrom0_bm0(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=None, savetag=savetag)

# c=0 chromaticity, no basemap error.
def run_set_gen_binwise_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=None, savetag='')

def plot_set_binwise_chromflat_bm0(*Npolys):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=None)

# 1.6e-2 chromaticity, no basemap error.
def run_set_gen_binwise_chromsmall_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=None, savetag=savetag)

def plot_set_binwise_chromsmall_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=None, savetag=savetag)

def plot_binwise_chi_sq_bic_chromsmall_bm0(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=None, savetag=savetag)

# 3.4e-2 chromaticity, no basemap error.
def run_set_gen_binwise_chrom_bm0(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=None, savetag=savetag)

def plot_set_binwise_chrom_bm0(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=None, savetag=savetag)

def plot_binwise_chi_sq_bic_chrom_bm0(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=None, savetag=savetag)

# Achromatic, 5% basemap error.
def run_set_gen_binwise_chrom0_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=5, savetag=savetag)

def plot_set_binwise_chrom0_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=5, savetag=savetag)

# c=0 chromaticity, 5% basemap error.
def run_set_gen_binwise_chromflat_bm5(*Npolys):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=0, basemap_err=5, savetag='')

def plot_set_binwise_chromflat_bm5(*Npolys):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=5)

# 1.6e-2 chromaticity, 5% basemap error.
def run_set_gen_binwise_chromsmall_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=5, savetag=savetag)

def plot_set_binwise_chromsmall_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

def plot_binwise_chi_sq_bic_chromsmall_bm5(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=5, savetag=savetag)

# 3.4e-2 chromaticity, 5% basemap error.
def run_set_gen_binwise_chrom_bm5(*Npolys, savetag=''):
    for Npoly in Npolys:
        gen_binwise_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=5, savetag=savetag)

def plot_set_binwise_chrom_bm5(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_binwise_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=5, savetag=savetag)

def plot_binwise_chi_sq_bic_chrom_bm5(*Npolys, savetag=None):
    plot_binwise_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=5, savetag=savetag)


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
    ax[0].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
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

def plot_binwise_chrom_pair(Nant=7, Npolys=(5,6), chromstr='1.6e-02', basemap_err=None, savetag=None):
    runstrs    = [f"Nant<{Nant}>_Npoly<{Npoly}>" for Npoly in Npolys]
    savestr    = f"Nant<{Nant}>_Npoly<{Npolys[0]}><{Npolys[1]}>"
    if chromstr is not None:
        runstrs[0] += f"_chrom<{chromstr}>"
        runstrs[1] += f"_chrom<{chromstr}>"
        savestr    += f"_chrom<{chromstr}>"
    else:
        runstrs[0] += f"_achrom"
        runstrs[1] += f"_achrom"
        savestr    += f"_achrom"
    if basemap_err is not None:
        runstrs[0] += f"_bm<{basemap_err}>"
        runstrs[1] += f"_bm<{basemap_err}>"
        savestr   += f"_bm<{basemap_err}>"
    if savetag is not None:
        runstrs[0] += savetag
        runstrs[1] += savetag
        savestr    += savetag
    print("loading from", runstrs)
    mcmcChains = [np.load('saves/Binwise/'+runstr+'_mcmcChain.npy') for runstr in runstrs]
    datas      = [np.load('saves/Binwise/'+runstr+'_data.npy') for runstr in runstrs]
    dataerrs   = [np.load('saves/Binwise/'+runstr+'_dataerr.npy') for runstr in runstrs]

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChains[0]))), size=1000)
    samples_mcmcChain = mcmcChains[0][idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)

    fig, ax = plt.subplots(2, 2, figsize=(5,3), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax[0,0].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0,0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0,0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[1,0].set_xlabel("Frequency [MHz]")
    ax[0,0].set_ylabel("21-cm Temperature [K]")
    ax[0,0].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[0,0].legend(loc='lower right')
    top_plot_spacing = 0.02
    ax00_ymax = np.max(a00mean_mcmc+2*a00std_mcmc)*alm2temp + top_plot_spacing
    ax00_ymin = np.min(a00mean_mcmc-2*a00std_mcmc)*alm2temp - top_plot_spacing

    mat_A_dummy = FM.generate_dummy_mat_A(OBS.nuarr, Ntau=1, lmod=32)
    mod = FM.generate_binwise_cm21_forward_model(nuarr=OBS.nuarr, observation_mat=mat_A_dummy, Npoly=Npolys[0])
    ax[1,0].axhline(y=0, linestyle=':', color='k')
    ax[1,0].errorbar(OBS.nuarr, mod(np.mean(mcmcChains[0], axis=0))-datas[0], dataerrs[0], fmt='.', color='k', ms=2)
    ax[1,0].set_ylabel(r"$T_\mathrm{res}$ [K]")
    bottom_plot_spacing = 0.01
    ax10_ymax = np.max(mod(np.mean(mcmcChains[0], axis=0))-datas[0]) + bottom_plot_spacing
    ax10_ymin = np.min(mod(np.mean(mcmcChains[0], axis=0))-datas[0]) - bottom_plot_spacing

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChains[1]))), size=1000)
    samples_mcmcChain = mcmcChains[1][idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)

    ax[0,1].plot(OBS.nuarr, cm21_a00*alm2temp, linestyle=':', color='k')
    ax[0,1].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none'
    )
    ax[0,1].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[1,1].set_xlabel("Frequency [MHz]")
    ax[0,1].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    top_plot_spacing = 0.02
    ax01_ymax = np.max(a00mean_mcmc+2*a00std_mcmc)*alm2temp + top_plot_spacing
    ax01_ymin = np.min(a00mean_mcmc-2*a00std_mcmc)*alm2temp - top_plot_spacing

    mat_A_dummy = FM.generate_dummy_mat_A(OBS.nuarr, Ntau=1, lmod=32)
    mod = FM.generate_binwise_cm21_forward_model(nuarr=OBS.nuarr, observation_mat=mat_A_dummy, Npoly=Npolys[1])
    ax[1,1].axhline(y=0, linestyle=':', color='k')
    ax[1,1].errorbar(OBS.nuarr, mod(np.mean(mcmcChains[1], axis=0))-datas[1], dataerrs[1], fmt='.', color='k', ms=2)
    ax11_ymax = np.max(mod(np.mean(mcmcChains[1], axis=0))-datas[1]) + bottom_plot_spacing
    ax11_ymin = np.min(mod(np.mean(mcmcChains[1], axis=0))-datas[1]) - bottom_plot_spacing

    ax[0,0].set_ylim([min(ax00_ymin, ax01_ymin), max(ax00_ymax, ax01_ymax)])
    ax[0,1].set_ylim([min(ax00_ymin, ax01_ymin), max(ax00_ymax, ax01_ymax)])
    ax[1,0].set_ylim([min(ax10_ymin, ax11_ymin), max(ax10_ymax, ax11_ymax)])
    ax[1,1].set_ylim([min(ax10_ymin, ax11_ymin), max(ax10_ymax, ax11_ymax)])

    # Turn off the y axis ticklabels for the right plots.
    ax[0,1].set_yticklabels([])
    ax[1,1].set_yticklabels([])

    fig.tight_layout()
    if savetag is not None:
        plt.savefig(f"fig/Binwise/bw_"+savestr+savetag+".pdf")
        plt.savefig(f"fig/Binwise/bw_"+savestr+savetag+".png")
    plt.show()

def plot_showcase_binwise():
    """
    The final four-panel figure to showcase all binwise modelling/fitting.
    Involves a figure showing chrom0, chrom small, chrom large and the BIC plots
    for each of these all on one subplot.
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Create subplots for first three panels (top-left, top-right, bottom-left)
    for i in range(3):
        row = i // 2
        col = i % 2
        # Create nested GridSpec for the panel with ratio 3:1
        nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[row, col],
                                                    height_ratios=[3, 1])
        ax_top = fig.add_subplot(nested_gs[0])
        ax_bottom = fig.add_subplot(nested_gs[1])
        
        # Store axes for later use
        if i == 0:
            ax_tl_top, ax_tl_bottom = ax_top, ax_bottom
        elif i == 1:
            ax_tr_top, ax_tr_bottom = ax_top, ax_bottom
        else:
            ax_bl_top, ax_bl_bottom = ax_top, ax_bottom

    # Create single subplot for bottom-right panel
    ax_br = fig.add_subplot(gs[1, 1])
    Npoly1, Npoly2, Npoly3 = 3, 5, 6
    runstr_tl = construct_runstr(Nant=7, Npoly=Npoly1, chromstr=None, basemap_err=None)
    runstr_tr = construct_runstr(Nant=7, Npoly=Npoly2, chromstr='1.6e-02', basemap_err=None)
    runstr_bl = construct_runstr(Nant=7, Npoly=Npoly3, chromstr='3.4e-02', basemap_err=None)

    def construct_plot(ax_top, ax_bottom, runstr, Npoly):
        # Load data
        mcmcChain = np.load('saves/Binwise/'+runstr+'_mcmcChain.npy')
        data      = np.load('saves/Binwise/'+runstr+'_data.npy')
        dataerr   = np.load('saves/Binwise/'+runstr+'_dataerr.npy')

        # Calculate contours and fid line.
        cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
        cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)
        idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
        samples_mcmcChain = mcmcChain[idx_mcmcChain]
        samples_mcmcChain = samples_mcmcChain[:,-3:]
        a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
        a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
        a00std_mcmc  = np.std(a00list_mcmc, axis=0)
        # Plot
        ax_top.plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
        ax_top.fill_between(
            OBS.nuarr,
            (a00mean_mcmc-a00std_mcmc)*alm2temp, 
            (a00mean_mcmc+a00std_mcmc)*alm2temp,
            color='C1',
            alpha=0.8,
            edgecolor='none',
            label="inferred"
        )
        ax_top.fill_between(
            OBS.nuarr,
            (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
            (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
            color='C1',
            alpha=0.4,
            edgecolor='none'
        )
        ax_top.set_ylabel("21-cm Temperature [K]")
        ax_top.set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
        ax_top.set_xticklabels([])
        ax_bottom.set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
        ax_bottom.set_xlabel("Frequency [MHz]")

        mat_A_dummy = FM.generate_dummy_mat_A(OBS.nuarr, Ntau=1, lmod=32)
        mod = FM.generate_binwise_cm21_forward_model(nuarr=OBS.nuarr, observation_mat=mat_A_dummy, Npoly=Npoly)
        ax_bottom.axhline(y=0, linestyle=':', color='k')
        ax_bottom.errorbar(OBS.nuarr, mod(np.mean(mcmcChain, axis=0))-data, dataerr, fmt='.', color='k', ms=2)
        ax_bottom.set_ylabel(r"$T_\mathrm{res}$ [K]")

    construct_plot(ax_tl_top, ax_tl_bottom, runstr_tl, Npoly1)
    construct_plot(ax_tr_top, ax_tr_bottom, runstr_tr, Npoly2)
    construct_plot(ax_bl_top, ax_bl_bottom, runstr_bl, Npoly3)
    ax_tl_top.legend(loc='lower right')


    # Make bottom-right subplot.
    Npolys = [3,4,5,6,7]
    runstrs_chrom0     = [construct_runstr(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=None) for Npoly in Npolys]
    runstrs_chromsmall = [construct_runstr(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=None) for Npoly in Npolys]
    runstrs_chrom      = [construct_runstr(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=None) for Npoly in Npolys]
    bics_chrom0     = [np.load('saves/Binwise/'+runstr+'_bic.npy') for runstr in runstrs_chrom0]
    bics_chromsmall = [np.load('saves/Binwise/'+runstr+'_bic.npy') for runstr in runstrs_chromsmall]
    bics_chrom      = [np.load('saves/Binwise/'+runstr+'_bic.npy') for runstr in runstrs_chrom]
    ax_br.semilogy(Npolys, bics_chrom0, color='C0', linestyle='-', marker='o', label='achromatic')
    ax_br.semilogy(Npolys, bics_chromsmall, color='C1', linestyle='-', marker='s', label='c=1.6e-02')
    ax_br.semilogy(Npolys, bics_chrom, color='C2', linestyle='-', marker='^', label='c=3.4e-02')
    ax_br.set_ylabel("Model BIC")
    ax_br.set_xlabel("$N_\mathrm{poly}$")
    ax_br.legend(loc='upper right')
    ax_br.set_xticks(ticks=Npolys, labels=Npolys, minor=False)
    ax_br.set_xlim([Npolys[0], Npolys[-1]])

    # Set spacing between subplots
    fig.tight_layout()
    plt.savefig("fig/Binwise/showcase_binwise.pdf")
    plt.show()

    # CORNER PLOT.
    mcmcChain_tl  = np.load('saves/Binwise/'+runstr_tl+'_mcmcChain.npy')
    mcmcChain_tr  = np.load('saves/Binwise/'+runstr_tr+'_mcmcChain.npy')
    mcmcChain_bl  = np.load('saves/Binwise/'+runstr_bl+'_mcmcChain.npy')

    param_labels = [r'$A_{21}$', r'$\nu_{21}$', r'$\Delta$']
    param_tags   = ['A', 'nu', 'delta']
    tagged_chain_tl = {tag: value for tag, value in zip(param_tags, mcmcChain_tl[:,-3:].transpose())}
    tagged_chain_tr = {tag: value for tag, value in zip(param_tags, mcmcChain_tr[:,-3:].transpose())}
    tagged_chain_bl = {tag: value for tag, value in zip(param_tags, mcmcChain_bl[:,-3:].transpose())}
    tagged_chain_tl['config'] = {'name' : ''}
    tagged_chain_tr['config'] = {'name' : ''}
    tagged_chain_bl['config'] = {'name' : ''}
    cornerplot = AxesCornerPlot(tagged_chain_bl, tagged_chain_tr, tagged_chain_tl, 
                                labels=param_labels, param_truths=OBS.cm21_params)
    cornerplot.set_figurepad(0.15)
    cornerfig = cornerplot.get_figure()
    cornerfig.savefig("fig/Binwise/showcase_binwise_corner.pdf")
    plt.show()



def plot_ml_chrom_cornerpair(Nant1=7, Nant2=7, Npoly1=7, Npoly2=7, chromstr1=None, chromstr2=None, basemap_err1=None, basemap_err2=None, savetag=None):
    runstr1 = construct_runstr(Nant1, Npoly1, chromstr1, basemap_err1)
    runstr2 = construct_runstr(Nant2, Npoly2, chromstr2, basemap_err2)
    print("loading from", runstr1, "and", runstr2, sep='\n')

    mcmcChain1  = np.load('saves/MLmod/'+runstr1+'_mcmcChain.npy')
    mcmcChain2  = np.load('saves/MLmod/'+runstr2+'_mcmcChain.npy')
    cm21_chain1 = mcmcChain1[:,-3:]
    cm21_chain2 = mcmcChain2[:,-3:]

    param_labels = [r'$A_{21}$', r'$\nu_{21}$', r'$\Delta$']
    param_tags   = ['A', 'nu', 'delta']
    tagged_chain1 = {tag: value for tag, value in zip(param_tags, cm21_chain1.transpose())}
    tagged_chain2 = {tag: value for tag, value in zip(param_tags, cm21_chain2.transpose())}
    tagged_chain1['config'] = {'name' : ''}
    tagged_chain2['config'] = {'name' : ''}

    cornerplot = AxesCornerPlot(tagged_chain2, tagged_chain1, 
                                labels=param_labels, param_truths=OBS.cm21_params)
    cornerplot.set_xticks("A", [-0.4, -0.3, -0.2, -0.1])
    #cornerplot.set_xticks("nu", [-0.4, -0.3, -0.2])
    cornerplot.set_figurepad(0.15)
    f = cornerplot.get_figure()
    
    '''
    c = ChainConsumer()
    c.add_chain(mcmcChain2[:,-3:], parameters=[r'$A_{21}$', r'$nu_{21}$', r'$\Delta$'], name='')
    c.add_chain(mcmcChain1[:,-3:], name='')
    f = c.plotter.plot(truth=[*OBS.cm21_params])'''
    if savetag is not None:
        f.savefig(f"fig/MLmod/pairplots/ml_"+runstr1+savetag+"_corner.pdf")
        f.savefig(f"fig/MLmod/pairplots/ml_"+runstr1+savetag+"_corner.png")
    plt.show()


def plot_showcase_ml_corner():
    """
    Final corner plot for showcasing ML modelling, featuring the 21-cm
    posteriors for achromatic case and c=3.4e-2 both with 10% foreground 
    correction errors, as well as c=3.4e-2 with 20% foreground correction 
    errors.
    """
    Nants = [7,7,7]
    Npolys = [3,3,4]
    chromstrs = [None, '3.4e-02', '3.4e-02']
    basemap_errs = [10, 10, 20]
    runstrs = [construct_runstr(Nants[i], Npolys[i], chromstrs[i], basemap_errs[i]) for i in range(3)]
    print("loading from", runstrs)

    mcmcChains = [np.load('saves/MLmod/'+runstr+'_mcmcChain.npy') for runstr in runstrs]
    cm21_chains = [chain[:,-3:] for chain in mcmcChains]

    param_labels = [r'$A_{21}$', r'$\nu_{21}$', r'$\Delta$']
    param_tags   = ['A', 'nu', 'delta']
    tagged_chains = [{tag: value for tag, value in zip(param_tags, chain.transpose())} for chain in cm21_chains]
    for i in range(3):
        tagged_chains[i]['config'] = {'name' : '', 'shade_alpha' : 0.1}
    
    cornerplot = AxesCornerPlot(tagged_chains[2], tagged_chains[1], tagged_chains[0], 
                                labels=param_labels, param_truths=OBS.cm21_params,
                                plotter_kwargs={'figsize':"COLUMN"})
    cornerplot.set_figurepad(0.15)
    f = cornerplot.get_figure()
    f.savefig("fig/MLmod/showcase_ml_corner.pdf")
    plt.show()




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
    nontrivial_obs_memopt_missing_modes(Npoly=Npoly, lats=ant_LUT[Nant], chrom=chrom, basemap_err=basemap_err, err_type='idx', mcmc=True, mcmc_pos=startpos, steps=100000, burn_in=60000, plotml=False)

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
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='0.0e+00', basemap_err=0, savetag=savetag)
    
def plot_ml_chi_sq_bic_chromflat_bm0(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='0.0e+00', basemap_err=0, savetag=savetag)

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

# Achromatic, 10% basemap error.
def run_set_gen_ml_chrom0_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=None, basemap_err=10)

def plot_set_ml_chrom0_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr=None, basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chrom0_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr=None, basemap_err=10, savetag=savetag)

# 1.6e-2 chromaticity, 10% basemap error.
def run_set_gen_ml_chromsmall_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=1.6e-2, basemap_err=10)

def plot_set_ml_chromsmall_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='1.6e-02', basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chromsmall_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='1.6e-02', basemap_err=10, savetag=savetag)

# 3.4e-2 chromaticity, 10% basemap error.
def run_set_gen_ml_chrom_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=10)

def plot_set_ml_chrom_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chrom_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=10, savetag=savetag)

# 3.4e-2 chromaticity, 20% basemap error.
def run_set_gen_ml_chrom_bm20(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=3.4e-2, basemap_err=20)

def plot_set_ml_chrom_bm20(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='3.4e-02', basemap_err=20, savetag=savetag)

def plot_ml_chi_sq_bic_chrom_bm20(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='3.4e-02', basemap_err=20, savetag=savetag)

# 5.2e-2 chromaticity, 10% basemap error.
def run_set_gen_ml_chromlarge_bm10(*Npolys):
    for Npoly in Npolys:
        gen_ml_chrom(Nant=7, Npoly=Npoly, chrom=5.2e-2, basemap_err=10)

def plot_set_ml_chromlarge_bm10(*Npolys, savetag=None):
    for Npoly in Npolys:
        plot_ml_chrom(Nant=7, Npoly=Npoly, chromstr='5.2e-02', basemap_err=10, savetag=savetag)

def plot_ml_chi_sq_bic_chromlarge_bm10(*Npolys, savetag=None):
    plot_ml_chi_sq_bic(Nant=7, Npolys=Npolys, chromstr='5.2e-02', basemap_err=10, savetag=savetag)


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
    if savetag is not None:
        f.savefig(f"fig/MLmod/ml_"+runstr+savetag+"_corner.pdf")
        f.savefig(f"fig/MLmod/ml_"+runstr+savetag+"_corner.png")
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
    ax[0].set_ylabel(r"21-cm Temperature [K]")
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

def plot_ml_chrom_pair(Nant1=7, Nant2=7, Npoly1=7, Npoly2=7, chromstr1=None, chromstr2=None, basemap_err1=None, basemap_err2=None, savetag=None):
    runstr1 = construct_runstr(Nant1, Npoly1, chromstr1, basemap_err1)
    runstr2 = construct_runstr(Nant2, Npoly2, chromstr2, basemap_err2)
    print("loading from", runstr1, "and", runstr2, sep='\n')


    fig, ax = plt.subplots(2, 2, figsize=(6,4), sharex=True, gridspec_kw={'height_ratios':[3,1]})

    mcmcChain = np.load('saves/MLmod/'+runstr1+'_mcmcChain.npy')
    residuals = np.load('saves/MLmod/'+runstr1+'_modres.npy')
    data      = np.load('saves/MLmod/'+runstr1+'_data.npy')
    dataerr   = np.load('saves/MLmod/'+runstr1+'_dataerr.npy')
    fid_a00   = np.load("saves/MLmod/"+runstr1+"_fid_a00.npy")
    rec_a00   = np.load("saves/MLmod/"+runstr1+"_rec_a00.npy")
    a00_error = np.load("saves/MLmod/"+runstr1+"_rec_a00_err.npy")
    
    # Calculate number of timeseries data points per antenna to reshape the data
    # arrays.
    Nfreq = len(OBS.nuarr)
    Ntau  = int(len(data) / (Nfreq*Nant1))
    data  = np.reshape(data, (Nfreq, Nant1, Ntau))
    dataerr   = np.reshape(dataerr, (Nfreq, Nant1, Ntau))
    residuals = np.reshape(residuals, (Nfreq, Nant1, Ntau))

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)
    ax[0,0].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0,0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0,0].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[0,0].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[1,0].set_xlabel("Frequency [MHz]")
    ax[0,0].set_ylabel(r"21-cm Temperature [K]")
    ax[0,0].legend()
    top_plot_spacing = 0.02
    ax00_ymax = np.max(a00mean_mcmc+2*a00std_mcmc)*alm2temp + top_plot_spacing
    ax00_ymin = np.min(a00mean_mcmc-2*a00std_mcmc)*alm2temp - top_plot_spacing

    ax[1,0].axhline(y=0, linestyle=':', color='k')
    ax[1,0].errorbar(OBS.nuarr, (rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp, a00_error*alm2temp, fmt='.', color='k', ms=2)
    ax[1,0].axhline(0, linestyle=':', color='k')
    ax[1,0].set_ylabel(r"$\hat{T}_\mathrm{mon}-\mathcal{M}$ [K]")
    bottom_plot_spacing = 0.01
    ax10_ymax = np.max(rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp + np.max(a00_error*alm2temp) + bottom_plot_spacing
    ax10_ymin = np.min(rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp - np.max(a00_error*alm2temp) - bottom_plot_spacing


    mcmcChain = np.load('saves/MLmod/'+runstr2+'_mcmcChain.npy')
    residuals = np.load('saves/MLmod/'+runstr2+'_modres.npy')
    data      = np.load('saves/MLmod/'+runstr2+'_data.npy')
    dataerr   = np.load('saves/MLmod/'+runstr2+'_dataerr.npy')
    fid_a00   = np.load("saves/MLmod/"+runstr2+"_fid_a00.npy")
    rec_a00   = np.load("saves/MLmod/"+runstr2+"_rec_a00.npy")
    a00_error = np.load("saves/MLmod/"+runstr2+"_rec_a00_err.npy")

    # Calculate number of timeseries data points per antenna to reshape the data
    # arrays.
    Nfreq = len(OBS.nuarr)
    Ntau  = int(len(data) / (Nfreq*Nant2))
    data  = np.reshape(data, (Nfreq, Nant2, Ntau))
    dataerr   = np.reshape(dataerr, (Nfreq, Nant2, Ntau))
    residuals = np.reshape(residuals, (Nfreq, Nant2, Ntau))

    # Plot inferred signal.
    cm21_a00_mod = lambda nuarr, theta: np.sqrt(4*np.pi)*SM.cm21_globalT(nuarr, *theta)
    cm21_a00 = cm21_a00_mod(OBS.nuarr, theta=OBS.cm21_params)

    idx_mcmcChain = np.random.choice(a=list(range(len(mcmcChain))), size=1000)
    samples_mcmcChain = mcmcChain[idx_mcmcChain]
    samples_mcmcChain = samples_mcmcChain[:,-3:]
    a00list_mcmc = [cm21_a00_mod(OBS.nuarr, theta) for theta in samples_mcmcChain]
    a00mean_mcmc = np.mean(a00list_mcmc, axis=0)
    a00std_mcmc  = np.std(a00list_mcmc, axis=0)
    ax[0,1].plot(OBS.nuarr, cm21_a00*alm2temp, label='fiducial', linestyle=':', color='k')
    ax[0,1].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.8,
        edgecolor='none',
        label="inferred"
    )
    ax[0,1].fill_between(
        OBS.nuarr,
        (a00mean_mcmc-2*a00std_mcmc)*alm2temp, 
        (a00mean_mcmc+2*a00std_mcmc)*alm2temp,
        color='C1',
        alpha=0.4,
        edgecolor='none'
    )
    ax[0,1].set_xlim([OBS.nuarr[0], OBS.nuarr[-1]])
    ax[1,1].set_xlabel("Frequency [MHz]")
    ax01_ymax = np.max(a00mean_mcmc+2*a00std_mcmc)*alm2temp + top_plot_spacing
    ax01_ymin = np.min(a00mean_mcmc-2*a00std_mcmc)*alm2temp - top_plot_spacing

    ax[1,1].axhline(y=0, linestyle=':', color='k')
    ax[1,1].errorbar(OBS.nuarr, (rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp, a00_error*alm2temp, fmt='.', color='k', ms=2)
    ax[1,1].axhline(0, linestyle=':', color='k')
    ax11_ymax = np.max(rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp + np.max(a00_error*alm2temp) + bottom_plot_spacing
    ax11_ymin = np.min(rec_a00-fg_cm21_polymod(OBS.nuarr, *np.mean(mcmcChain, axis=0)))*alm2temp - np.max(a00_error*alm2temp) - bottom_plot_spacing

    ax[0,0].set_ylim([min(ax00_ymin, ax01_ymin), max(ax00_ymax, ax01_ymax)])
    ax[0,1].set_ylim([min(ax00_ymin, ax01_ymin), max(ax00_ymax, ax01_ymax)])
    ax[1,0].set_ylim([min(ax10_ymin, ax11_ymin), max(ax10_ymax, ax11_ymax)])
    ax[1,1].set_ylim([min(ax10_ymin, ax11_ymin), max(ax10_ymax, ax11_ymax)])
    # Turn off the y axis ticklabels for the right plots.
    ax[0,1].set_yticklabels([])
    ax[1,1].set_yticklabels([])

    fig.tight_layout()
    if savetag is not None:
        plt.savefig(f"fig/MLmod/pairplots/ml_"+runstr1+"and"+runstr2+savetag+".pdf")
        plt.savefig(f"fig/MLmod/pairplots/ml_"+runstr1+"and"+runstr2+savetag+".png")
    plt.show()


def plot_ml_chrom_cornerpair(Nant1=7, Nant2=7, Npoly1=7, Npoly2=7, chromstr1=None, chromstr2=None, basemap_err1=None, basemap_err2=None, savetag=None):
    runstr1 = construct_runstr(Nant1, Npoly1, chromstr1, basemap_err1)
    runstr2 = construct_runstr(Nant2, Npoly2, chromstr2, basemap_err2)
    print("loading from", runstr1, "and", runstr2, sep='\n')

    mcmcChain1  = np.load('saves/MLmod/'+runstr1+'_mcmcChain.npy')
    mcmcChain2  = np.load('saves/MLmod/'+runstr2+'_mcmcChain.npy')
    cm21_chain1 = mcmcChain1[:,-3:]
    cm21_chain2 = mcmcChain2[:,-3:]

    param_labels = [r'$A_{21}$', r'$\nu_{21}$', r'$\Delta$']
    param_tags   = ['A', 'nu', 'delta']
    tagged_chain1 = {tag: value for tag, value in zip(param_tags, cm21_chain1.transpose())}
    tagged_chain2 = {tag: value for tag, value in zip(param_tags, cm21_chain2.transpose())}
    tagged_chain1['config'] = {'name' : ''}
    tagged_chain2['config'] = {'name' : ''}

    cornerplot = AxesCornerPlot(tagged_chain2, tagged_chain1, 
                                labels=param_labels, param_truths=OBS.cm21_params)
    cornerplot.set_xticks("A", [-0.4, -0.3, -0.2, -0.1])
    #cornerplot.set_xticks("nu", [-0.4, -0.3, -0.2])
    cornerplot.set_figurepad(0.15)
    f = cornerplot.get_figure()
    
    '''
    c = ChainConsumer()
    c.add_chain(mcmcChain2[:,-3:], parameters=[r'$A_{21}$', r'$nu_{21}$', r'$\Delta$'], name='')
    c.add_chain(mcmcChain1[:,-3:], name='')
    f = c.plotter.plot(truth=[*OBS.cm21_params])'''
    if savetag is not None:
        f.savefig(f"fig/MLmod/pairplots/ml_"+runstr1+savetag+"_corner.pdf")
        f.savefig(f"fig/MLmod/pairplots/ml_"+runstr1+savetag+"_corner.png")
    plt.show()


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
    run_set_gen_ml_chrom0_bm0(3,4,5,6,7)
    run_set_gen_ml_chromsmall_bm0(3,4,5,6,7)
    run_set_gen_ml_chrom_bm0(3,4,5,6,7)

    run_set_gen_ml_chrom0_bm5(3,4,5,6,7)
    run_set_gen_ml_chromsmall_bm5(3,4,5,6,7)
    run_set_gen_ml_chrom_bm5(3,4,5,6,7)

    run_set_gen_ml_chrom0_bm10(3,4,5,6,7)
    run_set_gen_ml_chromsmall_bm10(3,4,5,6,7)
    run_set_gen_ml_chrom_bm10(3,4,5,6,7)

def plot_all_ml():
    """
    Batch plot all run_set_gen_ml functions in the script for Npoly=3, ..., 7.
    """

    plot_set_ml_chrom0_bm0(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chrom0_bm0(3,4,5,6,7,savetag='')
    plot_set_ml_chromsmall_bm0(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chromsmall_bm0(3,4,5,6,7,savetag='')
    plot_set_ml_chrom_bm0(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chrom_bm0(3,4,5,6,7,savetag='')

    plot_set_ml_chrom0_bm5(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chrom0_bm5(3,4,5,6,7,savetag='')
    plot_set_ml_chromsmall_bm5(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chromsmall_bm5(3,4,5,6,7,savetag='')
    plot_set_ml_chrom_bm5(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chrom_bm5(3,4,5,6,7,savetag='')

    plot_set_ml_chrom0_bm10(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chrom0_bm10(3,4,5,6,7,savetag='')
    plot_set_ml_chromsmall_bm10(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chromsmall_bm10(3,4,5,6,7,savetag='')
    plot_set_ml_chrom_bm10(3,4,5,6,7,savetag='')
    plot_ml_chi_sq_bic_chrom_bm10(3,4,5,6,7,savetag='')

def run_all_binwise():
    """
    Batch run all run_set_gen_binwise functions in the script for Npoly=3, ..., 7.
    """
    run_set_gen_binwise_chrom0_bm0(3, 4, 5, 6, 7)
    run_set_gen_binwise_chromflat_bm0(3, 4, 5, 6, 7)
    run_set_gen_binwise_chromsmall_bm0(3, 4, 5, 6, 7)
    run_set_gen_binwise_chrom_bm0(3, 4, 5, 6, 7)
    run_set_gen_binwise_chrom0_bm5(3, 4, 5, 6, 7)
    run_set_gen_binwise_chromflat_bm5(3, 4, 5, 6, 7)
    run_set_gen_binwise_chromsmall_bm5(3, 4, 5, 6, 7)
    run_set_gen_binwise_chrom_bm5(3, 4, 5, 6, 7)
