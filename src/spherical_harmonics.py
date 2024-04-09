"""
Classes and functions that deal with spherical harmonics, including: 

RealSphericalHarmonics
    A class to translate between Healpy's complex alm and a basis of
    real valued alm and spherical harmonics.

calc_spherical_harmonic_matrix, calc_inv_spherical_harmonic_matrix
    Code to calculate the real valued Y matrix of spherical harmonics and the 
    inverse matrix to go from pixel data to alm. 
    Both are linear transforms for this discrete case.

The class contains private test methods and calc_spherical_harmonic_matrix has a test function.
"""
import healpy
import numpy as np
import numpy.random as npr

class RealSphericalHarmonics:
    """
    Class for translating complex spherical harmonics from Healpy into
    real spherical harmonics.

    Because the connection between real and complex Ylm and alm has the same
    structure, this should work for both the Ylm and the alm.

    For real alm, I'll use an ordering by l and then m, so that the vector of alm
    is indexed as i = l(l+1) + m  and data is stored as
    [ (0,0), (1,-1), (1,0), (1,1),...,(l,-l),(l,-l+1),...,(l,l-1),(l,l),...]

    TODO:
    - Work out if there's a more efficient way of doing this than looping through things.
      The indexing is hard for me to parse, so a loop is the good brute force way, but
      probably there's a better way.

    """

    def __init__(self):
        """
        No real set up needed
        """
        pass

    def get_size(self, lmax):
        """
        get size of vector required to store lmax

        Need to be a little careful with sums over l since lmax is
        the physical quantity not a vector length
        """

        vsize = 0
        for l in range(lmax+1):
            vsize += (2*l+1)

        return vsize

    def get_idx(self, l, m):
        """
        Find index from (l,m)
        """
        return l * (l+1) + m

    def get_lmax(self, vsize):
        """
        Get lmax required for a given vector size.
        """
        lmax = np.sqrt(vsize) - 1
        if abs(int(lmax)-lmax) > 0.00001:
            err = "Invalid vector size! Does not correspond to any lmax"
            raise ValueError(err)
        return int(lmax)


    def get_lm(self, idx):
        """
        Find (l,m) from vector index
        """

        l = int(np.sqrt(idx))
        ll = l * (l+1)
        m = idx - ll

        return l, m

    def real2ComplexALM(self, almr):
        """
        Convert a set of real alm to the Healpy ordered complex alm
        """

        lmax = int(np.sqrt(len(almr))) - 1
        assert len(almr) == self.get_size(lmax)
        #print("lmax 1=", lmax)

        almc = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)
        for l in range(lmax+1):
            for m in range(l+1):
                #print("(l,m)=",l,m)
                idxc = healpy.Alm.getidx(lmax, l, m)
                idxc_m = healpy.Alm.getidx(lmax, l, -m)
                if m == 0:
                    almc_real = almr[self.get_idx(l,m)]
                    almc[idxc] = almc_real
                else:
                    almc_real = almr[self.get_idx(l,m)] / np.sqrt(2.0)
                    almc_imag = almr[self.get_idx(l,-m)] / np.sqrt(2.0)
                    almc[idxc] = almc_real + almc_imag * 1.0j
                    #almc[idxc_m] = (-1)**m * almc_real - almc_imag * 1.0j
                #print(almc_real)

        return almc

    def complex2RealALM(self, almc):
        """
        Convert a set of Healpy ordered complex alm to real alm

        There are (lmax+1)(lmax+2)/2 components in the healpy vector
        """

        lmax = int(np.sqrt(len(almc) * 2 - 1)) - 1
        assert len(almc) == healpy.Alm.getsize(lmax)
        #print("lmax 2=", lmax)

        almr = np.zeros(self.get_size(lmax))
        for l in range(lmax+1):
            for m in range(l+1):
                #print("(l,m)=",l,m)
                idxc = healpy.Alm.getidx(lmax, l, m)
                idxr_p = self.get_idx(l,m)
                idxr_m = self.get_idx(l,-m)

                if m == 0:
                    almr[idxr_p] = almc[idxc]
                else:
                    almr[idxr_p] = np.real(almc[idxc]) * np.sqrt(2.0)
                    almr[idxr_m] = np.imag(almc[idxc]) * np.sqrt(2.0)
                #print(almc[idxc])

        return almr

    def _test_translation(self, lmax=5):
        """
        test translation back and forth from real to complex to real
        """
        result_flag = True
        tol = 1.0e-4

        almr = np.arange(self.get_size(lmax)) + 1
        almc = self.real2ComplexALM(almr)
        almr_recover = self.complex2RealALM(almc)

        #test start and end values are within tol of each other
        for i in range(len(almr)):
            #print(almr[i], almr_recover[i])
            value = (abs(float(almr[i]) - almr_recover[i]) < tol)
            result_flag = (result_flag and value)

        print(result_flag)

        return almr, almc, almr_recover

    def _test_indexing_functions(self, lmax = 10):
        """
        test get_idx and get_lm
        """

        result_flag = True

        #test can go back and forth from (l,m) to idx consistently
        for l in range(lmax+1):
            for m in range(-l,l):
                idx = self.get_idx(l,m)
                lp, mp = self.get_lm(idx)
                value = (l==lp) and (m==mp)
                #print(value, idx, l, m, lp, mp)
                result_flag = (result_flag and value)

        return result_flag


def calc_spherical_harmonic_matrix(nside=8, lmax=20):
    """
    construct the matrix for the spherical harmonics
    Y_ij = Y_{l_jm_j}(r_i)

    The end matrix is npix x nalm in size where
    npix is the number of pixels from the nside healpy sphere
    nalm is the number of real valued alm

    map = sum_lm Y * alm

    """
    RS = RealSphericalHarmonics()

    #So then the matrix for the spherical harmonics
    npix = healpy.pixelfunc.nside2npix(nside)
    nalm = healpy.Alm.getsize(lmax)
    print("calc_spherical_harmonic_matrix npix, nalm :", npix, nalm)

    # Construct the complex version of the Ylm
    Y_matrix_real = np.zeros((npix, nalm))
    Y_matrix_imag = np.zeros((npix, nalm))
    Y_complex = np.zeros((npix, nalm),dtype=np.complex128)

    for j in range(nalm):
        l, m = healpy.Alm.getlm(lmax, j)

        #Use Healpy functions to get the Ylm by using alm=1 for a single alm
        alm = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)

        #first get the real part of Ylm by using only one alm = 1
        alm = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)
        alm[healpy.Alm.getidx(lmax, l, m)] = 1.0
        if m == 0:
            Y_matrix_real[:,j] = healpy.sphtfunc.alm2map(alm, nside=nside)
        else:
            Y_matrix_real[:,j] = healpy.sphtfunc.alm2map(alm, nside=nside) / 2.0

        #now get the imaginary part by using only one alm = j
        alm = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)
        alm[healpy.Alm.getidx(lmax, l, m)] = 1.0j
        Y_matrix_imag[:,j] = healpy.sphtfunc.alm2map(alm, nside=nside) / 2.0

        Y_complex[:,j] = Y_matrix_real[:,j] + Y_matrix_imag[:,j] * 1.0j

    # convert from complex Ylm to real Ylm by looping over pixels
    # basically treating each pixel as having a set of Y_lm(indx) to convert
    nalm_real = RS.get_size(lmax)
    Y_real = np.zeros((npix, nalm_real))
    for i in range(npix):
        Y_real[i,:] = RS.complex2RealALM(Y_complex[i,:])

    return Y_real


def calc_inv_spherical_harmonic_matrix(nside=8, lmax=20):
    """
    construct the inverse matrix for the spherical harmonics
    Y_ij = Y_{l_jm_j}^*(r_i)

    The end matrix is nalm x npix in size where
    npix is the number of pixels from the nside healpy sphere
    nalm is the number of real valued alm

    alm = sum_pix invY * map

    """
    RS = RealSphericalHarmonics()

    #So then the matrix for the spherical harmonics
    npix = healpy.pixelfunc.nside2npix(nside)
    nalm = healpy.Alm.getsize(lmax)
    print(npix, nalm)

    #inverse matrix for Y is really just a rescaled version of Y.T
    Y = calc_spherical_harmonic_matrix(nside=nside, lmax=lmax)
    invY = Y.T * (4*np.pi) / npix

    return invY


def _test_reconstruction(nside=8, lmax=20, use_random=True, show_images=True):
    """
    Run a test to see if reconstruction works

    From either a random or GSM input map
    1. calculate the alm from the input map
    2. using only the l<lmax modes create a filtered map by transform
    3. using only l<lmax modes reconstruct test map from Y_matrix
    4. compare the filtererd and reconstructed maps to see if successful

    The filtered and reconstructed maps should agree perfectly. The input map
    and the filtered/reconstructed maps won't completely agree because there is
    extra information in the input map that isn't captured in the filtering.
    """
    from pygdsm import GlobalSkyModel2016

    #select the map to test against
    if use_random:
        # real testing needs a bandwidth limited map created by making random alm
        # of the right length.
        alm = np.zeros(healpy.Alm.getsize(lmax), dtype=np.complex128)
        alm_random = npr.normal(0.0,1.0,size=alm.shape) + npr.normal(0.0,1.0,size=alm.shape) *1.0j
        map_random = healpy.sphtfunc.alm2map(alm_random, nside=nside) #map reconstructed directly from alm
        map_input = map_random
    else:
        # Make a map at 100 MHz
        gsm_2016 = GlobalSkyModel2016(freq_unit='MHz', resolution='low')
        map100 = gsm_2016.generate(100)  #store output healpy data
        gsm_2016.view(logged=True)

        #Down grade map resolution
        map100_lo = healpy.pixelfunc.ud_grade(map100, nside_out=nside)
        map_input = map100_lo

    # try using this to reconstruct the map
    alm_ref = healpy.sphtfunc.map2alm(map_input, lmax=lmax, use_weights=False)
    map_filtered = healpy.sphtfunc.alm2map(alm_ref, nside=nside, pixwin=False) #map reconstructed directly from alm

    #Now create the reconstructed test map from the Y_matrix
    RS = RealSphericalHarmonics()
    alm_ref_real = RS.Complex2RealALM(alm_ref)
    Y_real = calc_spherical_harmonic_matrix(nside=nside, lmax=lmax)

    # reconstruct sky from real valued alm using real valued Y matrix
    map_test = Y_real @ alm_ref_real

    if show_images:
        #image the maps
        healpy.visufunc.mollview(map_input)
        healpy.visufunc.mollview(map_test)
        healpy.visufunc.mollview(map_filtered)

        #image the residuals
        healpy.visufunc.mollview(map_input-map_test)
        healpy.visufunc.mollview(map_filtered-map_test)
        healpy.visufunc.mollview(map_input-map_filtered)

    #verify that filtered and reconstructed maps agree
    result_flag = np.all(np.abs(map_filtered - map_test) < 1.0e-4)
    return result_flag
