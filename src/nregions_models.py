"""
Implementation of the Nregions models of Anstey et. al. (2021, 2023) and Pagano
et. al. (2024).
"""
import numpy as np
from numba import jit
from anstey.generate import T_CMB

def pix_forward_model_pl(powers, nuarr, base_map, masks):
    """
    Return the Nregions model in Anstey et. al. (2021) evaluated at the
    frequencies nuarr, given a list of masks, a base map at 408 MHz and a list
    of powers to raise each masked region to.

    Parameters
    ----------
    powers, masks : (Nregions,) array, (Nregions, Npix) list of arrays
        The power to raise each masked region to, and the masks themselves.
    nuarr : (Nfreq,) array
        The frequencies to evaluate at.
    base_map : (Npix,) array
        The base map at 408 MHz to use.
    
    Returns
    -------
    temperatures : (len(nuarr)*len(base_map),) array
    """
    # Check inputs.
    if len(powers) != len(masks):
        raise ValueError("should have the same number of powers as masks.")
    if np.shape(masks)[1] != len(base_map):
        raise ValueError("masks should have the same number of pixels as the base temperature map.")
    if len(np.shape(powers)) != 1 or len(np.shape(masks)) != 2:
        raise ValueError("bad number of input dimensions.")
    
    result = np.zeros(shape=len(nuarr)*len(base_map))
    for power, mask in zip(powers, masks):
        masked_basemap = mask*(base_map - T_CMB)
        single_term = [masked_basemap*(nu/408)**(-power) for nu in nuarr]
        single_term = np.array(single_term).flatten()
        result += single_term
    return result + T_CMB


def genopt_pix_forward_model_pl(nuarr, base_map, masks):
    """
    Return the Nregions model in Anstey et. al. (2021) evaluated at the
    frequencies nuarr, given a list of masks, a base map at 408 MHz and a list
    of powers to raise each masked region to.

    Parameters
    ----------
    powers, masks : (Nregions,) array, (Nregions, Npix) list of arrays
        The power to raise each masked region to, and the masks themselves.
    nuarr : (Nfreq,) array
        The frequencies to evaluate at.
    base_map : (Npix,) array
        The base map at 408 MHz to use.
    
    Returns
    -------
    temperatures : (len(nuarr)*len(base_map),) array
    """
    # Decide the data vector length.
    block_len = len(base_map)
    
    @jit
    def model(powers):
        result = np.zeros(shape=len(nuarr)*len(base_map))
        for power, mask in zip(powers, masks):
            masked_basemap = mask*(base_map - T_CMB)
            for i in range(len(nuarr)):
                result[i*block_len:(i+1)*block_len] = masked_basemap*(nuarr[i]/408)**(-power)
        return result + T_CMB
    return model
