'''
Code that generates the spectral indices of the 2008 PyGDSM pixel by pixel,
following the method of Anstey et. al. 2021 [2010.09644].
'''

import numpy as np
from pygdsm import GlobalSkyModel

def main():
    T_CMB = 2.725
    sky   = GlobalSkyModel()
    T_230, T_408 = sky.generate([230, 408])
    indexes = -np.log((T_230-T_CMB)/(T_408-T_CMB)) / np.log(230/408)
    np.save('/Users/yordani/Documents/boosted_compass/matrix-observer/anstey/indexes', 
            np.array([T_408, indexes]))

if __name__ == '__main__':
    main()