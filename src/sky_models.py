"""
Models for the alm describing the sky.

Models should return a vector of alm with structure
  mock = (alms(nu1), alms(nu2), ...)
 where the alms correspond to real values alms with lmax
 for some set of frequencies nu.

"""
import healpy
import numpy as np
import os

import src.sky_models as SM
from src.spherical_harmonics import RealSphericalHarmonics
RS = RealSphericalHarmonics()
if os.uname()[1]=='yordan-XPS-15-9560':
    from pygdsm import GlobalSkyModel16 as GlobalSkyModel2016
    ROOT = '/home/yordan/Documents/boosted-compass/matrix-observer'
else:
    from pygdsm import GlobalSkyModel2016
    ROOT = '/Users/yordani/Documents/boosted_compass/matrix-observer'
from numba import jit
from anstey.generate import T_CMB, gen_err_gsma

import src.spherical_harmonics as SH
from src.blockmat import BlockMatrix, BlockVector
from src.spherical_harmonics import RealSphericalHarmonics, calc_spherical_harmonic_matrix

RS = RealSphericalHarmonics()

def add_noise(temps, dnu=None, Ntau=None, t_int=None, dtB=None, seed=124):
    """
    Generate Gaussian radiometer noise for the passed temperature vector. Either 
    use dnu (frequency bin width in MHz), Ntau (number of time bins) and t_int 
    (total integration time in hours) to calculate dtB for each bin, or pass dtB
    directly.

    Returns the tuple: (data, data noise covar matrix)
    The noise covariance is assumed to be a diagonal matrix, with the noise
    variance along the diagonals.
    """
    np.random.seed(seed)
    #if t_int is passed, use it to calculate dtB
    if t_int is not None:
        dt  = 3600*t_int/(Ntau)  #interval in seconds
        dtB = dt*1e+6*dnu        #convert interval to Hertz and multiply by B

        
    temperr = temps/np.sqrt(dtB)
    
    if isinstance(temps, BlockVector):
        covar = BlockMatrix(np.diag(temperr.vector**2), mode='as-is', nblock=temps.nblock)
        noisy_temps = np.random.normal(temps.vector, np.abs(temperr.vector))
        return BlockVector(noisy_temps, mode='as-is', nblock=temps.nblock), covar

    covar = np.diag(temperr**2)
    return np.random.normal(temps, np.abs(temperr)), covar

def add_noise_uniform(temps, err, seed=123):
    """
    Generate uniform noise for the passed temperature vector.

    Returns the tuple: (data, data noise covar matrix)
    The noise covariance is assumed to be a diagonal matrix, with the noise
    variance along the diagonals.
    """
    np.random.seed(seed)
    if isinstance(temps, BlockVector):
        covar = BlockMatrix(np.diag([err**2]*temps.vec_len), mode='as-is', nblock=temps.nblock)
        noisy_temps = np.random.normal(temps.vector, np.abs(err))
        return BlockVector(noisy_temps, mode='as-is', nblock=temps.nblock), covar

    covar = np.diag([err**2]*len(temps))
    return np.random.normal(temps, np.abs(err)), covar

@jit
def cm21_globalT(nu, A=-0.2, nu0=80.0, dnu = 5.0):
    """
    Return the Gaussian 21-cm monopole model Tmon(nu) for an
    array of frequencies nu.
    """
    chi = (nu - nu0) / dnu
    Tmon = A * np.exp(-0.5 * chi * chi)
    return Tmon

def cm21_dipoleT(nu, A=-0.2, nu0=80.0, dnu = 5.0, beta = 1.2e-3):
    """
    Return the Gaussian 21-cm monopole and dipole frequency
    components for an array of frequencies nu.
    
        T_dip(nu) = beta * (Tmon - nu * dTmon/dnu) * n.nstar 
                  = beta * F(nu) * n.nstar

    Here capture beta * F(nu) and leave angular bit for elsewhere

    returns: Tmon, Tdip
    """
    chi = (nu - nu0) / dnu
    Tmon = cm21_globalT(nu, A, nu0, dnu)
    dTmon = -chi / dnu * Tmon

    Tdip = beta * (Tmon - nu * dTmon)

    return Tmon, Tdip

def cm21_gauss_mondip_alm(nu, lmax, params = None):
    """
    Given the lmax for real alms build a model consisting of the
    monopole + dipole from the 21cm signal as a Gaussian.

    Create copies of the alm for each frequency to
    return mock = (alms(nu1), alms(nu2), ...)
    """
    if params is None:
        A = -0.2
        nu0 = 80.0
        dnu = 5.0
        beta = 1.2e-3
        alpha = 0.2*np.pi
        delta = 0.5*np.pi
    else:
        A, nu0, dnu, beta, alpha, delta = params

    #build the alm shape vector
    alm = np.zeros(RS.get_size(lmax))

    #mock monopole signal
    alm[0] = np.sqrt(4*np.pi)

    if beta != 0:
        #mock dipole signal
        norm = np.sqrt(4.0 * np.pi / 3.0)
        a10  = norm * np.cos(delta)
        a11p = norm * np.sin(delta) * np.cos(alpha)
        a11m = norm * np.sin(delta) * np.sin(alpha)
        alm[1] += a11m
        alm[2] += a10
        alm[3] += a11p

    #calculate the monopole and dipole values
    tb, tb_dip = cm21_dipoleT(nu, A, nu0, dnu, beta)

    Nalm = len(alm)
    Nfreq = len(nu)
    mock2 = np.ones([Nfreq, Nalm])
    mock2 = np.multiply(mock2, alm).T
    mock2[0,:] = np.multiply(mock2[0,:], tb)
    mock2[1:4,:] = np.multiply(mock2[1:4,:], tb_dip)
    mock2 = mock2.T
    mock = mock2.reshape(Nfreq * Nalm)
    return mock

def cm21_gauss_mon_alm(nu, lmax, params = None):
    """
    Given the lmax for real alms build a model consisting of the
    monopole from the 21cm signal as a Gaussian.

    Create copies of the alm for each frequency to
    return mock = (alms(nu1), alms(nu2), ...)
    """
    if params is None:
        A = -0.2
        nu0 = 80.0
        dnu = 5.0
    else:
        A, nu0, dnu = params
    return cm21_gauss_mondip_alm(nu, lmax, params=[A, nu0, dnu, 0, 0, 0])

def foreground_mondip_alm(nu, lmax, params = None):
    """
    Given the lmax for real alms build a model consisting of the
    monopole + dipole for the foregrounds as power laws

    Create copies of the alm for each frequency to
    return mock = (alms(nu1), alms(nu2), ...)

    Following Ignatov+ (2023)
    """
    if params is None:
        Amono = 5382.0
        alpha_mono = 2.726
        Adip = 3100.0
        alpha_dip = 2.555
        alpha = 0.2*np.pi
        delta = 0.5*np.pi
    else:
        Amono, alpha_dip, Adip, alpha_dip, alpha, delta = params

    #build the alm shape vector
    alm = np.zeros(RS.get_size(lmax))

    #mock monopole signal
    alm[0] = np.sqrt(4*np.pi)

    #mock dipole signal
    norm = np.sqrt(4.0 * np.pi / 3.0)
    a10  = norm * np.cos(delta)
    a11p = norm * np.sin(delta) * np.cos(alpha)
    a11m = norm * np.sin(delta) * np.sin(alpha)
    alm[1] = a11m
    alm[2] = a10
    alm[3] = a11p

    #calculate the monopole value
    tb = Amono * np.power(nu / 60.0, -alpha_mono)

    #calculate the dipole value
    tb_dip = Adip * np.power(nu / 60.0, -alpha_dip)

    #map this to the alm_nu space
    # this can be vectorised to avoid the loop, I think
    r = []
    for i in range(len(nu)):
        alm_use = alm.copy()
        alm_use[0] *= tb[i]
        alm_use[1:4] *= tb_dip[i]
        r.append(alm_use)
    mock = np.concatenate(r)

    return mock

def foreground_gdsm_alm(nu, lmax=40, nside=None, map=False):
    """
    Calculate the vector of real alm for the 2016 GDSM evaluated
    at the frequenc(y/ies) nu. Returns the alms in a flat array 
    (alms(nu1), alms(nu2), ...). 
    
    If nside is given, the GDSM map is first up/downgraded to this nside.
    This helps speed up conversion to spherical harmonics, but may lose 
    information.

    If map is True, also returns the alm represented as a series of
    healpix maps. The resulting nside will match the nside argument,
    or will be the same as the GDSM's high-resolution map.
    """
    #are we dealing with multiple frequencies or not
    try:
        len(nu)
    except:
        nu = [nu]
    
    #generate the map
    gdsm_2016 = GlobalSkyModel2016(freq_unit='MHz', resolution='hi')
    gdsm_map = [gdsm_2016.generate(freq) for freq in nu]
    
    #degrade the foreground map to the size we want
    if nside is not None:
        gdsm_map = healpy.pixelfunc.ud_grade(gdsm_map, nside_out=nside)
    nside = healpy.npix2nside(np.shape(gdsm_map)[-1])

    #convert to (real) alm, dealing with both multi and single freq cases
    map_alm = [healpy.sphtfunc.map2alm(m, lmax=lmax) for m in gdsm_map]
    map_real_alm = np.array([RS.complex2RealALM(alms) for alms in map_alm])

    #convert the alm back to healpix maps
    if map:
        reconstucted_map = [healpy.sphtfunc.alm2map(alms, nside=nside) for alms in map_alm]
        return map_real_alm.flatten(), reconstucted_map
    return map_real_alm.flatten()


def foreground_gdsm_galcut_alm(nu, lmax=40, nside=None, map=False):
    """
    Calculate the vector of real alm for the 2016 GDSM evaluated
    at the frequenc(y/ies) nu. Returns the alms in a flat array 
    (alms(nu1), alms(nu2), ...). 
    
    If nside is given, the GDSM map is first up/downgraded to this nside.
    This helps speed up conversion to spherical harmonics, but may lose 
    information.

    If map is True, also returns the alm represented as a series of
    healpix maps. The resulting nside will match the nside argument,
    or will be the same as the GDSM's high-resolution map.

    This function cuts the galaxy out (setting values to the mean map value), 
    and sets all negative temperatures to zero too.
    """
    #are we dealing with multiple frequencies or not
    try:
        len(nu)
        multifreq = True
    except:
        multifreq = False
    
    #generate the map
    gdsm_2016 = GlobalSkyModel2016(freq_unit='MHz', resolution='hi')
    gdsm_map = gdsm_2016.generate(nu)

    negs = np.where(gdsm_map < 0)
    gdsm_map[negs] = 0
    gdsm_map[int(len(gdsm_map)*0.45):int(len(gdsm_map)*0.55)] = np.mean(gdsm_map)
    
    #degrade the foreground map to the size we want
    if nside is not None:
        gdsm_map = healpy.pixelfunc.ud_grade(gdsm_map, nside_out=nside)
    nside = healpy.npix2nside(np.shape(gdsm_map)[-1])

    #convert to (real) alm, dealing with both multi and single freq cases
    if multifreq:
        map_alm = [healpy.sphtfunc.map2alm(m, lmax=lmax) for m in gdsm_map]
        map_real_alm = np.array([RS.complex2RealALM(alms) for alms in map_alm])
    else:
        map_alm = healpy.sphtfunc.map2alm(gdsm_map, lmax=lmax)
        map_real_alm = RS.complex2RealALM(map_alm)

    #convert the alm back to healpix maps
    if map:
        if multifreq:
            reconstucted_map = [healpy.sphtfunc.alm2map(alms, nside=nside) for alms in map_alm]
        else:
            reconstucted_map = healpy.sphtfunc.alm2map(map_alm, nside=nside)
        return map_real_alm.flatten(), reconstucted_map
    return map_real_alm.flatten()


def _gsma_indexes_to_alm(nu, T_408, indexes, lmax=40, nside=None, map=False, original_map=False, use_mat_Y=False, meancorr=False, delta=None):
    #are we dealing with multiple frequencies or not
    try:
        len(nu)
    except:
        nu = [nu]
    nside = healpy.npix2nside(len(T_408))

    #generate the map
    gsma_map = [(T_408 - T_CMB)*(freq/408)**(-indexes) + T_CMB for freq in nu]

    #perform a mean-correction
    if delta is not None and meancorr:
        print('correcting for mean')
        gsma_map = [(m-T_CMB)*np.exp(-(delta*np.log(freq/408))**2/2) + T_CMB for m, freq in zip(gsma_map, nu)]

    #convert to (real) alm, dealing with both multi and single freq cases
    if use_mat_Y:
        mat_Y = calc_spherical_harmonic_matrix(nside=nside, lmax=lmax, try_loading=True)
        inv_mat_Y = np.linalg.pinv(mat_Y)
        map_real_alm = np.array([inv_mat_Y @ m for m in gsma_map])
    elif not use_mat_Y:
        map_alm = [healpy.sphtfunc.map2alm(m, lmax=lmax) for m in gsma_map]
        map_real_alm = np.array([RS.complex2RealALM(alms) for alms in map_alm])

    #convert the alm back to healpix maps
    if map:
        reconstucted_map = [healpy.sphtfunc.alm2map(alms, nside=nside) for alms in map_alm]
        return map_real_alm.flatten(), np.array(reconstucted_map)
    elif original_map:
        return map_real_alm.flatten(), np.array(gsma_map)
    return map_real_alm.flatten()

def foreground_gsma_alm(nu, lmax=40, nside=None, map=False):
    '''
    An extrapolation of the GDSM sky back to the 21-cm frequency range as used
    in Anstey et. al. 2021 (arXiv:2010.09644).

    Calculate the vector of real alm for the GSMA evaluated
    at the frequenc(y/ies) nu. Returns the alms in a flat array 
    (alms(nu1), alms(nu2), ...). 
    
    If nside is given, the GSMA map is first up/downgraded to this nside.
    This helps speed up conversion to spherical harmonics, but may lose 
    information.

    If map is True, also returns the alm represented as a series of
    healpix maps. The resulting nside will match the nside argument,
    or will be equal to 512, the native resolution of the GSMA.
    '''
    #load the gsma indexes
    try:
        T_408, indexes = np.load(ROOT+'/anstey/indexes.npy')
    except:
        raise Exception("Indexes for the Anstey sky have not been "\
                        +"generated.")
    return _gsma_indexes_to_alm(nu, T_408=T_408, indexes=indexes, lmax=lmax, 
                                nside=nside, map=map)

def basemap_err_to_delta(percent_err, ref_freq=70):
    """
    Roughly calculate the delta error in GSMA power law index required for a 
    given basemap error, taking the basemap error AS A PERCENTAGE.
    """
    if percent_err is None:
        return None
    return np.log(percent_err*1e-2 + 1) / np.log(408/ref_freq)

def foreground_gsma_alm_nsidelo(nu, lmax=32, nside=None, map=False, original_map=False, use_mat_Y=False, const_idx=False, delta=None, err_type='idx', seed=123, meancorr=False):
    '''
    An extrapolation of the GDSM sky back to the 21-cm frequency range as used
    in Anstey et. al. 2021 (arXiv:2010.09644). This version uses the same
    strategy, but generates an nside=16 map instead.

    Calculate the vector of real alm for the GSMA evaluated
    at the frequenc(y/ies) nu. Returns the alms in a flat array 
    (alms(nu1), alms(nu2), ...). 
    
    If nside is given, the GSMA map is first up/downgraded to this nside.
    This helps speed up conversion to spherical harmonics, but may lose 
    information.

    If map is True, also returns the alm represented as a series of
    healpix maps. The resulting nside will match the nside argument,
    or will be equal to 16, the native resolution of the GSMA lo.

    If original_map is True, will also return the GSMA map before it's been 
    converted to alm (ignored if map=True).

    If use_mat_Y is True, calculates/uses the spherical harmonic matrix Y.

    If const_index is true, will scale the Haslam map back with a constant power
    law index of -2.5.

    GENERATING DIFFERENT FOREGROUND INSTANCES:

    If delta is not None, will generate a map with errors induced in it. How it 
    does this depends on err_type:
        'inx' - delta is width of gaussian random error added to the standard 
                 GSMA indexes.
        'bm1' - Adds delta fractional error to the T_230 GSMA basemap but not 
                 the T_408 basemap.
        'bm2' - Adds delta fractional error to both basemaps.
        'bm'  - Only used if const_idx=True. This just adds percentage errors to
                the T_408 basemap.

    Seed is the random seed to do the above.
    '''
    #load the gsma indexes
    if nside is None:
        nside=32
    try:
        T_408, indexes = np.load(ROOT+f'/anstey/indexes_{nside}.npy')
        if const_idx:
            indexes = 2.7*np.ones_like(indexes)
            if delta is not None and err_type=='bm':
                np.random.seed(seed)
                T_408 = np.random.normal(loc=T_408, scale=T_408*delta)
    except:
        raise Exception(f"Indexes for the Anstey sky nside={nside} have not been "\
                        +"generated.")
    if delta is not None:
        if seed is None:
            seed=123
        np.random.seed(seed)
        
        if err_type=='idx':
            indexes = np.random.normal(loc=indexes, scale=delta)
        elif err_type=='bm1':
            T_408, indexes = gen_err_gsma(nside_out=nside, delta=delta, 
                                          one_basemap=True, seed=seed)
        elif err_type=='bm2':
            T_408, indexes = gen_err_gsma(nside_out=nside, delta=delta, 
                                          one_basemap=False, seed=seed)
        elif err_type=='bm':  # Dealt with above.
            pass
        else:
            raise ValueError("Invalid err_type passed.")

    return _gsma_indexes_to_alm(nu, T_408=T_408, indexes=indexes, lmax=lmax, 
                                map=map, original_map=original_map, use_mat_Y=use_mat_Y, meancorr=meancorr, delta=delta)

def foreground_gsma_nsidelo(nu, nside=None):
    """
    Simply returns the degraded GSMA map in pixel space for the frequenc(y/ies)
    nu.
    """
    _, gsma_map = foreground_gsma_alm_nsidelo(nu, original_map=True, nside=nside)
    if gsma_map.shape[0] == 1:
        return gsma_map.flatten()
    return gsma_map

def haslam_scaled_map(nu, nside=None):
    """
    Simply returns the degraded Haslam map scaled back with a fixed power law
    index of -2.5.
    """
    _, hl_map = foreground_gsma_alm_nsidelo(nu, original_map=True, 
                                            nside=nside, const_idx=True)
    if hl_map.shape[0] == 1:
        return hl_map.flatten()
    return hl_map


def gsma_corr(lmod, lmax, nside, nuarr, bmerr, ref_freq=230):
    """
    Compute and return the GSMA error instance alm mean correction and covariance
    matrix, truncated as requested. Will first check to see if the relevant 
    correction has not already been generated, in which case it will be loaded
    instead.
    """
    Nlmod = RS.get_size(lmod)
    while True:
        try:
            alm_cov  = np.load("saves/gsma_corr_nside<{}>_lmax<{}>_bmerr<{}>_cov.npy".format(nside, lmax, bmerr))
            alm_cov  = alm_cov[:,Nlmod:,Nlmod:]
            alm_mean = np.load("saves/gsma_corr_nside<{}>_lmax<{}>_bmerr<{}>_mean.npy".format(nside, lmax, bmerr))
            alm_mean = alm_mean[:,Nlmod:]
            alm_mean = alm_mean.flatten()
            s = "Loaded mean and covar correction for nside<{}>_lmax<{}>_bmerr<{}>"
            print(s.format(nside, lmax, bmerr))
            return alm_mean, BlockMatrix(mat=alm_cov)
        except:
            s = "Failed loading mean and covar correction for nside<{}>_lmax<{}>_bmerr<{}>, generating instead."
            print(s.format(nside, lmax, bmerr))

            gsma_maps = SM.foreground_gsma_nsidelo(nu=nuarr, nside=nside)
            delta = SM.basemap_err_to_delta(percent_err=bmerr, ref_freq=ref_freq)

            temp_cov_blocks = []
            temp_mean_blocks = []

            for nu, gsma_map in zip(nuarr, gsma_maps):
                sigma_T   = delta * np.log(408/nu)

                exponents = np.exp(2*sigma_T**2) - np.exp(sigma_T**2)
                temp_cov_block_diag = exponents*(gsma_map - T_CMB)**2
                temp_cov_blocks.append(np.diag(temp_cov_block_diag))
                temp_mean_block = (gsma_map - T_CMB) * np.exp(sigma_T**2/2) + T_CMB
                temp_mean_blocks.append(temp_mean_block)

            inv_Y = np.linalg.pinv(SH.calc_spherical_harmonic_matrix(nside=nside, lmax=lmax))
            alm_mean = [inv_Y @ temp_mean_block for temp_mean_block in temp_mean_blocks]
            alm_cov  = [inv_Y @ temp_cov_block @ inv_Y.T for temp_cov_block in temp_cov_blocks]
            np.save("saves/gsma_corr_nside<{}>_lmax<{}>_bmerr<{}>_cov.npy".format(nside, lmax, bmerr), np.array(alm_cov))
            np.save("saves/gsma_corr_nside<{}>_lmax<{}>_bmerr<{}>_mean.npy".format(nside, lmax, bmerr), np.array(alm_mean))
    