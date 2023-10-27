"""
Models for the alm describing the sky.

Models should return a vector of alm with structure
  mock = (alms(nu1), alms(nu2), ...)
 where the alms correspond to real values alms with lmax
 for some set of frequencies nu.

"""
import healpy
import numpy as np
from pygdsm import GlobalSkyModel2016

from src.spherical_harmonics import RealSphericalHarmonics

RS = RealSphericalHarmonics()

def add_noise(temps, dnu=None, Ntau=None, t_int=None, dtB=None, seed=123):
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
    covar = np.diag(temperr**2)
    return np.random.normal(temps, temperr), covar

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

def cm21_gauss_mondip_alm_pymc(nu, lmax, params = None):
    """
    Given the lmax for real alms build a model consisting of the
    monopole + dipole from the 21cm signal as a Gaussian.

    Create copies of the alm for each frequency to
    return mock = (alms(nu1), alms(nu2), ...)

    This version intended for testing with pymc.

    output is [Nfreq x Nalm, 1]

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

    # calculate monopole and dipole temperatures
    chi = (nu - nu0) / dnu
    Tmono = A * np.exp(-0.5 * chi * chi)
    dTmono = -chi / dnu * Tmono
    Tdip = beta * (Tmono - nu * dTmono)

    #calculate monopole and dipole coefficients
    a00 = np.sqrt(4*np.pi)
    norm = np.sqrt(4.0 * np.pi / 3.0)
    a10  = norm * np.cos(delta)
    a11p = norm * np.sin(delta) * np.cos(alpha)
    a11m = norm * np.sin(delta) * np.sin(alpha)

    # projection basis
    Nalm = RS.get_size(lmax)
    proj00 = np.zeros([1,Nalm])
    proj00[0,0] = 1
    proj11m = np.zeros([1,Nalm])
    proj11m[0,1] = 1
    proj10 = np.zeros([1,Nalm])
    proj10[0,2] = 1
    proj11p = np.zeros([1,Nalm])
    proj11p[0,3] = 1

    #outer product to create matrix
    mock = proj00.T * Tmono * a00
    mock += proj11m.T * Tdip * a11m
    mock += proj10.T * Tdip * a10
    mock += proj11p.T * Tdip * a11p

    mock = mock.T
    mock = mock.reshape([len(nu) * Nalm, 1])

    return mock

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
    or will be the same as the GDSM's low-resolution map.
    """
    #are we dealing with multiple frequencies or not
    try:
        len(nu)
        multifreq = True
    except:
        multifreq = False
    
    #generate the map
    gdsm_2016 = GlobalSkyModel2016(freq_unit='MHz', resolution='low')
    gdsm_map = gdsm_2016.generate(nu)
    
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






