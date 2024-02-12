import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import src.spherical_harmonics as SH

RS = SH.RealSphericalHarmonics()

def compare_estimate_to_reality(a_reality, a_estimate, ylm_mat=None, nside=None,
                                mat_P=None):
    """
    Plot various figures comparing the resonstructed map to the original map,
    and the residuals of the estimated alm.

    Parameters
    ----------
    a_reality, a_estimate
        alm vectors corresponding to the fiducial map and the estimated map
        respectively. a_reality should be as long or longer than a_estimate.
    ylm_mat
        Ylm matrix corresponding to a_reality. This is chopped down to use for 
        a_estimate too. If none is passed, one will be created.
    nside
        If a Ylm matrix is being created, must pass a value for its nside. If 
        a Ylm matrix is being passed, the nside argument is ignored.
    mat_P
        If the observation matrix is passed, plots the pointing directions in
        the sky reconstruction plot.
    """
    try:
        lmax = RS.get_lmax(len(a_reality))
        lmod = RS.get_lmax(len(a_estimate))
        if lmod > lmax:
            msg = 'fiducial alm vector must be as long or longer than the ' \
                + 'estimated alm vector.'
            raise ValueError(msg)
    except ValueError:
        raise ValueError('invalid alm vector length.')
    
    if ylm_mat is None:
        if nside is None:
            raise ValueError("if Ylm matrix isn't passed, nside should be.")
        ylm_mat = SH.calc_spherical_harmonic_matrix(nside=nside, lmax=lmax)
    nside = hp.npix2nside(len(ylm_mat[:,0]))
    
    no_modes = len(a_estimate)
    ylm_mat_mod = ylm_mat[:, :no_modes]

    hp.mollview(ylm_mat@a_reality, title=f"lmax={lmax} original map")
    plt.show()

    hp.mollview(ylm_mat_mod@a_estimate, title=f"lmod={lmod} reconstructed map")
    plt.show()

    hp.mollview(ylm_mat_mod@a_estimate - ylm_mat@a_reality, 
                title=f"lmax={lmax}, lmod={lmod} reconstructed map residuals")
    plt.show()

    hp.mollview(np.abs(ylm_mat_mod@a_estimate - ylm_mat@a_reality), 
                title=f"lmax={lmax}, lmod={lmod} reconstructed map abs(residuals)")
    
    if mat_P is not None:
        # Use np.argmax to find the column index of the first occurrence of 1 in each row.
        indxs = np.argmax(mat_P, axis=1).tolist()
        thetas, phis = hp.pix2ang(nside=nside, ipix=indxs)
        hp.projscatter(thetas, phis, alpha=0.4, s=3, color='r')
    plt.show()

    mode_residuals = abs(a_estimate[:no_modes] - a_reality[:no_modes])
    mode_residuals /= abs(a_reality[:no_modes])
    lmod_arr = list(range(0, lmod+1, 2))
    lmod_idx = [RS.get_idx(l=l, m=-l) for l in lmod_arr]
    plt.axhline(y=1, color='k', alpha=0.7)
    plt.semilogy(range(no_modes), mode_residuals, '.')
    plt.xticks(ticks=lmod_idx, labels=lmod_arr)
    plt.xlabel("spherical harmonic l number")
    plt.ylabel("|alm estimated - alm fiducial|/|alm fiducial|")
    plt.show()
    

def compare_reconstructions(a_reality, *a_estimates, labels=None, fmts=None, ylm_mat=None,
                            return_comparisons_fig=False):
    """
    Compare the errors in reconstructions of the original alm vector by plotting
    the residuals of the vectors and optionally a bar chart of the different temperature
    standard deviations from the fiducial map.

    Parameters
    ----------
    a_reality, a_estimates
        Fiducial alm vector and set of estimated alm vectors. Estimated vectors
        must be as long or shorter than the fiducial vector.
    labels
        List of labels for the different estimates.
    fmts
        List of scatter plot format arguments for the estimates.
    ylm_mat
        ylm matrix corresponding to a_reality. If passed, calculates the
        residuals of the reconstructed temperature maps for each of the
        estimates and produces a bar plot.
    return_comparisons_fig : bool
        If True, will return the figure comparing the alm to one another and to
        the fiducial alm.
    """
    no_modes_list = [len(a) for a in a_estimates]
    max_modes = np.max(no_modes_list)
    lmod_max = RS.get_lmax(max_modes)
    lmod_max_arr = list(range(0, lmod_max+1, 2))
    lmod_max_idx = [RS.get_idx(l=l, m=-l) for l in lmod_max_arr]

    if labels is None:
        labels = ['']*len(a_estimates)
    if fmts is None:
        fmts = ['.']*len(a_estimates)
    
    fig, ax = plt.subplots(1, 1)
    for a_estimate, label, fmt, no_modes in zip(a_estimates, labels, fmts, no_modes_list):
        mode_residuals = abs(a_estimate[:no_modes] - a_reality[:no_modes])
        mode_residuals /= abs(a_reality[:no_modes])
        ax.semilogy(range(no_modes), mode_residuals, fmt, label=label)
    ax.axhline(y=1, color='k', alpha=0.7)
    ax.legend()
    ax.set_xticks(ticks=lmod_max_idx, labels=lmod_max_arr)
    ax.set_xlabel("spherical harmonic l number")
    ax.set_ylabel("|alm estimated - alm fiducial|/|alm fiducial|")
    ax.set_ylim(1e-7, 1e+5)
    if return_comparisons_fig:
        return fig
    fig.show()
    plt.close('all')

    # Make a bar chart comparing the reconstruction error in rms temperature.
    if ylm_mat is not None:
        map_reality = ylm_mat @ a_reality
        ylm_mats_trunc = [ylm_mat[:,:len(a)] for a in a_estimates]
        map_estimates = [y_mat @ a for y_mat, a in zip(ylm_mats_trunc, a_estimates)]
        rms = [np.std(map_reality - map_est) for map_est in map_estimates]
        plt.bar(x=list(range(len(rms))), height=rms)
        plt.xticks(ticks=list(range(len(rms))), labels=labels)
        plt.ylabel("RMS temperature error [K]")
        plt.xticks(rotation=30)
        plt.show()
    
    if ylm_mat is not None:
        lmax = RS.get_lmax(len(a_reality))
        idxs = [RS.get_idx(l=l, m=l)+1 for l in range(lmax)]
        cutoff_realities = np.array([ylm_mat[:,:i] @ a_reality[:i] for i in idxs])
        for a, label, fmt in zip(a_estimates, labels, fmts):
            lmax_a = RS.get_lmax(len(a))
            idxs = [RS.get_idx(l=l, m=l)+1 for l in range(lmax_a)]
            cutoff_estimates = np.array([ylm_mat[:,:i] @ a[:i] for i in idxs])
            rms = np.std(cutoff_estimates-cutoff_realities[:len(idxs)], axis=1)
            plt.semilogy(list(range(lmax_a)), rms, fmt, label=label)
        plt.xlabel("spherical harmonic l number")
        plt.ylabel("RMS temperature residuals [K]")
        plt.legend()
        plt.show()