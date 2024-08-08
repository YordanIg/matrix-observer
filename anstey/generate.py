'''
Code that generates the spectral indices of the 2008 PyGDSM pixel by pixel,
following the method of Anstey et. al. 2021 [2010.09644].
'''
import os
if os.uname()[1]=='yordan-XPS-15-9560':
    ROOT = '/home/yordan/Documents/boosted-compass/matrix-observer'
else:
    ROOT = '/Users/yordani/Documents/boosted_compass/matrix-observer'
import numpy as np
from pygdsm import GlobalSkyModel
from healpy import ud_grade
T_CMB = 2.725

def main(nside_out=None):
    sky   = GlobalSkyModel()
    T_230, T_408 = sky.generate([230, 408])

    tag=''
    if nside_out is not None:
        T_230 = ud_grade(T_230, nside_out)
        T_408 = ud_grade(T_408, nside_out)
        tag=f'_{nside_out}'
    indexes = -np.log((T_230-T_CMB)/(T_408-T_CMB)) / np.log(230/408)
    np.save(ROOT+'/anstey/indexes'+tag, 
            np.array([T_408, indexes]))

def gen_err_gsma(nside_out, delta, one_basemap=True, seed=123):
    """
    Generate the GSMA with delta fractional errors in one or both basemaps.
    """
    sky = GlobalSkyModel()
    T_230, T_408 = sky.generate([230, 408])
    T_230 = ud_grade(T_230, nside_out)
    T_408 = ud_grade(T_408, nside_out)

    np.random.seed(seed)
    T_230 = np.random.normal(loc=T_230, scale=T_230*delta)
    if not one_basemap:
        T_408 = np.random.normal(loc=T_408, scale=T_408*delta)
    
    indexes = -np.log((T_230-T_CMB)/(T_408-T_CMB)) / np.log(230/408)
    return np.array([T_408, indexes])

def regen_all():
    nsides = [8, 16, 32, 64]
    main()
    for nside in nsides:
        main(nside)

if __name__ == '__main__':
    regen_all()
