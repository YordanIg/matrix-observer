"""
The entire forward-modelling formalism brought together.
"""
import coordinates as CO
import spherical_harmonics as SH
import beam_functions as BF

def caulculate_G():
    pass

def calculate_Y():
    pass

def calculate_B():
    pass

def calculate_A(
    
    nside, lmax, beam_use=beam_cos, norm_flag=True
    ):
    return caulculate_G() @ CO.calc_pointing_matrix(*pointing_coordinates, nside=32, pixels=False) @ SH.calc_spherical_harmonic_matrix(nside, lmax) @ BF.calc_beam_matrix(nside, lmax, beam_use=beam_cos, norm_flag)
    