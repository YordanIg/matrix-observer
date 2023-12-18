# matrix-observer

## Theory

We want to model the observations of a drift-scan dipole antenna in a CMB
mapmaking kind of way. For a binned timeseries data vector d, the model is

`d = GPYBa + n`

where `a` is a real vector of sky alm, `B` is a diagonal matrix of real beam 
alm, `Y` is a spherical harmonic matrix that maps between spherical harmonic 
space and pixel space, `P` is the pointing matrix, which encodes the observation strategy by placing a '1' in each row in the position corresponding to the pixel
being observed at a given time step. `G` is an averaging matrix that bins the 
time series data from length Nt to length `Ntau` < `Nt`. `n` is Gaussian random
noise. 

This has an analytic maximum-likelihood solution:

`W = inv( A.T inv(N) A ) A.T inv(N)`

`a_ml = W d`

where `W` is the maximum-likelihood estimator matrix.


## Modules

In the folder `/src`:

`sky_models.py`:
Functions for calculating the **spherical harmonic coefficient vectors** of the 
foregrounds and 21-cm signal, and for adding noise to the timeseries data.

`beam_functions.py`:
Functions for calculating the **beam matrix** for an azimuthally symmetric beam.

`spherical_harmonics.py`:
Class for translating between real and complex spherical harmonic vectors and 
functions to calculate the **spherical harmonic matrix**.

`coordinates.py`:
Converting between different coordinate systems and functions to calculate the 
**pointing matrix**.

`forward_model.py`:
Functions that call the different matrix builder modules to build the total
**forward modelling matrix** for different preset observation strategies.

`map_making.py`:
Function to calculate the **maximum-likelihood estimator matrix**.

`plotting.py`:
Useful plotting functions.

`powerlaw_regression.py`:
Implementing the powerlaw x PCA foreground/21-cm separation method.


## Usage

### Single-frequency case

Manually forward-modelling observations of the GSMA sky up to a given `nside` 
and `lmax` at `nu` MHz:

```
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import src.sky_models as SM
import src.beam_functions as BM
import src.spherical_harmonics as SH
import src.coordinates as CO

# generate sky alm
gsma_alm, gsma_map = SM.foreground_gsma_alm(nu, lmax, nside, map=True)

# visualise the fiducial sky
hp.mollview(gsma_map, title="GSMA sky")
plt.show()

# construct the beam matrix
narrow_cosbeam = lambda theta : BM.beam_cos(theta, theta0=0.8)
mat_B = BM.calc_beam_matrix(nside, lmax, beam_use=narrow_cosbeam)

# construct the spherical harmonic matrix (can precompute for given lmax, nside)
mat_Y = SH.calc_spherical_harmonic_matrix(nside, lmax)

# construct the observation matrix for multi-antenna drift-scan
coords_1 = CO.obs_zenith_drift_scan(lat_1, lon_1, 
            times=np.linspace(0, 24, 24, endpoint=False))
...
coords_n = CO.obs_zenith_drift_scan(lat_n, lon_n, 
            times=np.linspace(0, 24, 24, endpoint=False))
mat_P_multi_ant = CO.calc_pointing_matrix(coords_1, ..., coords_n, nside=32)

# optionally compute a binning matrix with Ntau bins
Nt = np.shape(mat_P_single_ant)[0]
mat_G = FM.calc_averaging_matrix(Ntau, Nt)

# compute the full observation matrix and forward model
mat_A = mat_G @ mat_P_single_ant @ mat_Y @ mat_B
timestream_data = mat_A @ gsma_alm

# add noise for noise covariance matrix N in a 1 MHz frequency bin and t_int 
# hours of total integration time.
timestream_noisy, mat_N = SM.add_noise(timestream_data, dnu=1, Ntau=Ntau, 
            t_int=None, seed=123)
```
