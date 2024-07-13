"""
Generate the figures (I think) will end up in the paper.
"""
from functools import partial

import matplotlib.pyplot as plt

import src.observing as OBS
import src.sky_models as SM
import src.beam_functions as BF
import src.forward_model as FM


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
    plt.axhline(y=0, linestyle=':', color='k')
    plt.plot(OBS.nuarr,fid_temp.vector-err05_temp.vector, '.', label='5%')
    plt.plot(OBS.nuarr,fid_temp.vector-err10_temp.vector, '.', label='10%')
    plt.plot(OBS.nuarr,fid_temp.vector-err15_temp.vector, '.', label='15%')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    print("Basemap errors propagated to temperature (monopole removed)")
    if save:
        plt.savefig("fig/basemap_err_monopole.pdf")
    else:
        plt.show()
    plt.close("all")

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
    plt.axhline(y=0, linestyle=':', color='k')
    plt.plot(OBS.nuarr,fid_temp.vector-err05_temp.vector, '.', label='5%')
    plt.plot(OBS.nuarr,fid_temp.vector-err10_temp.vector, '.', label='10%')
    plt.plot(OBS.nuarr,fid_temp.vector-err15_temp.vector, '.', label='15%')
    plt.legend()
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    print("Basemap errors propagated to temperature (monopole and dipole removed)")
    if save:
        plt.savefig("fig/basemap_err_mondip.pdf")
    else:
        plt.show()
    plt.close("all")


