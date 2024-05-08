'''
Code that generates the spectral indices of the 2008 PyGDSM pixel by pixel,
following the method of Anstey et. al. 2021 [2010.09644].
'''

import numpy as np
from pygdsm import GlobalSkyModel
from healpy import ud_grade
T_CMB = 0#2.725

def main(nside_out=None):

    sky   = GlobalSkyModel()
    T_230, T_408 = sky.generate([230, 408])

    tag=''
    if nside_out is not None:
        T_230 = ud_grade(T_230, nside_out)
        T_408 = ud_grade(T_408, nside_out)
        tag=f'_{nside_out}'
    indexes = -np.log((T_230-T_CMB)/(T_408-T_CMB)) / np.log(230/408)
    np.save('/Users/yordani/Documents/boosted_compass/matrix-observer/anstey/indexes'+tag, 
            np.array([T_408, indexes]))

if __name__ == '__main__':
    main()