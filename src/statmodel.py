"""
A copy of statmodel.py that deals with discrete models of the form 
p(data|theta) instead of y(data|theta, x). data is a multidimensional array.
"""
from multiprocessing import Pool
from time import time
import numpy as np
from scipy.optimize import minimize
import emcee

import matplotlib.pyplot as plt

######################
# Statistical Models #
######################

class StatModel:
    """
    Set up a model M(theta_true), such that it can be evaluated at arbitrary
    theta, and its Bayesian quantities may be calculated.
    
    To set up subclasses, pass the relevant parameters to the init method of
    this class, and redefine the evaluate method. If noisy data generation is
    required, this may also be defined as a method of the subclass.
    """
    def __init__(self, labels, lim, outdims, theta_true=None, priormean=None, 
                 priorcovar=None):
        """
        Parameters:
        ----------
        labels: string or list of strings
            Label(s) of the model parameters (useful for plotting).
        lim: 2-tuple or list of 2-tuples
            Limits encapsulating the parameter space of the model.
        outdims: tuple
            Shape of the model's output.
        theta_true: float or array-like, optional
            True value(s) of the model parameters.
        priormean: array-like, optional
            Mean of prior gaussian if using this feature.
        priorcovar: array-like, optional
            Covariance matrix of the prior if using this feature.
        """
        if theta_true is not None:  # NOTE: if theta_true not passed, no checks occur.
            # Number of parameters passed must be consistent.
            if len(theta_true)-len(labels):
                raise ValueError('length of theta_true and labels must match.')
            if len(theta_true)-len(lim):
                raise ValueError('length of theta_true and lim must match.')
            
            # Limits must be in ascending order.
            self.dims = len(theta_true)  # Dimensions of the parameter space.
            for i in range(self.dims):
                if len(lim[i]) != 2:
                    raise ValueError('limits must contain two values.')
                if lim[i][0] < theta_true[i] < lim[i][1]:
                    continue
                else:
                    raise ValueError('''limits must be in ascending order and 
                                        bound the fiducial value.''')
            self.theta_true = np.asarray(theta_true)
        
        self.labels     = np.asarray(labels)
        self.lim        = np.asarray(lim, dtype=object)
        self.dims       = len(self.labels)
        self.outdims    = outdims

        # If using gaussian prior.
        self.priormean  = priormean
        self.priorcovar = priorcovar
        if self.priormean is not None:
            dim = len(self.priormean)
            if np.any(np.array(np.shape(priorcovar))-dim):
                raise ValueError('Shape of the prior mean and covariance ' \
                                +'matrix must be consistent.')
            self.covardet   = np.linalg.det(priorcovar)
            self.covarinv   = np.linalg.inv(priorcovar)
        

    def evaluate(self, theta=None):
        """
        Evaluate the model for parameter values theta. If theta is
        unpassed, uses the fiducial model parameters.
        
        Returns:
        -------
        ndarray:
            Model-dependent multidimensional array of data.
        """
        raise NotImplementedError('Use a subclass of StatModel.')

    def _polychord_prior(self, phi):
        '''
        Only used in conjunction with PolyChord's weird prior function
        requirements. Take an array phi with values in the range [0, 1], and
        rescale them to match the prior ranges for each parameter.
        '''
        thetalower, thetaupper = np.array(list(zip(*self.lim)))
        return thetalower + (thetaupper - thetalower) * phi

    def log_uniform_prior(self, theta):
        """
        Evaluate the model's log prior for parameter values theta, assuming a 
        uniform prior. 
        """
        thetalower, thetaupper = np.array(list(zip(*self.lim)))
        if np.all(theta < thetaupper) and np.all(theta > thetalower):
            return 0.0
        return -np.inf
    
    def log_gauss_prior(self, theta):
        """
        Evaluate the model's log prior for parameter values theta, assuming a 
        gaussian prior. 
        """
        norm = np.sqrt((2*np.pi)**self.dims * self.covardet)
        innerprod = np.dot(theta-self.priormean, np.dot(self.covarinv, theta-self.priormean))
        return -innerprod/2 - np.log(norm)

    def log_likelihood(self, theta, d, derr):
        """
        Evaluate the model's log likelihood.

        Parameters:
        ----------
        theta: array-like
            Parameter values to evaluate at. Length must match dimensionality of
            the model.
        d, derr: array-like
            Multidimensional arrays of data and errors whose shape must match
            the shape of the model's output.
        """
        # Check inputs.
        # :: TODO actually check d and derr against self.outdims.
        logterm   = np.log(np.sqrt(2*np.pi)*derr)
        modelterm = (self.evaluate(theta) - d)/derr
        tosum     = -logterm - 0.5*modelterm**2
        return np.sum(tosum)

    def log_posterior(self, theta, d, derr):
        """
        Evaluate the model's log posterior for parameter values theta, given 
        data d with errors derr.

        Parameters:
        ----------
        theta: array-like
            Parameter values to evaluate at. Length must match dimensionality of
            the model.
        d, derr: array-like
            Multidimensional arrays of data and errors whose shape must match
            the shape of the model's output.
        """
        # Check inputs.
        # :: TODO actually check d and derr against self.outdims.
        lp = self.log_uniform_prior(theta)
        if self.priormean is not None:
            lp += self.log_gauss_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, d, derr)


class Polyn(StatModel):
    """
    Parameter inference on a polynomial of order 3, of form ax^2 + bx + c.
    The model's parameter space spans (-1,1) for each parameter.
    """
    def __init__(self, a, b, c, x):
        """
        Parameters:
        ----------
        a, b, c: floats
            Polymonial coefficients.
        x: array-like
            List of x positions to evaluate the model at.
        """
        super().__init__(
            theta_true = (a, b, c), 
            labels     = ('a', 'b', 'c'), 
            lim        = ((-1,1),(-1,1),(-1,1)),
            outdims    = len(x)
        )
        for i in range(self.outdims-1):
            if x[i] > x[i+1]:
                raise ValueError("x positions must be in ascending order.")
        self.x = x
    
    def evaluate(self, theta=None):
        """
        Evaluate the model for parameter values theta. If theta is
        unpassed, uses the fiducial model parameters.
        
        Returns:
        -------
        ndarray:
            Model-dependent multidimensional array of data.
        """
        if theta is None:
            theta = self.theta_true
        a, b, c = theta
        return a*self.x**2 + b*self.x + c
    
    def gen_noise(self, scale=0.1, seed=123):
        """
        Generate noisy data.

        Parameters:
        ----------
        scale: float, optional
            Errors in y will be Gaussian distributed with width given by scale.
        seed: int, optional
            Seeds the random generator.
        """
        # Unpack upper and lower limits.
        xlowers, xuppers = self.x[0], self.x[-1]

        # Generate data distributed in the range [0,1], then rescale it.
        np.random.seed(seed)
        xdata = np.random.rand(self.outdims)
        xdata = xlowers + (xuppers - xlowers)*xdata
        
        # Calculate the corresponding y data points and spread them.
        yerr  = np.array([scale]*self.outdims)
        ydata = self.evaluate()
        return np.random.normal(ydata, yerr), yerr
    


############
# Samplers #
############

class Sampling:
    '''
    A base class for inference.
    '''
    def __init__(self, model, d, derr):
        '''
        Parameters:
        ----------
        model: StatModel subclass
            The statistical model to use.
        d, derr: arrays
            Noisy data to use for inference.
        '''
        if not isinstance(model, StatModel):
            raise TypeError('model should be a subclass of StatModel.')
        
        if np.shape(d) != np.shape(derr):
            raise ValueError('lengths of data and its error should match.')
        
        self.model = model
        self.d     = d
        self.derr  = derr

    def inference(self, nwalkers=32, steps=5000, p0=None, spread=1e-4, 
            checkstep=None, tol=0.005, timer=True, progress=True, cpus=1,
            posterior_evals=False):
        """
        Use Emcee to infer the model parameters given noisy data, returning an
        emcee.EnsembleSampler object that has been run.

        Parameters
        ----------
        nwalkers, steps : int, int
            Number of walkers and number of steps taken for Emcee.
        p0 : array-like, optional
            Initial guess in parameter space. If none passed, uses the parameter
            values that maximise the likelihood.
        spread : float
            Walkers initialized in Gaussian ball with standard deviation spread.
        checkstep : int, optional
            Check convergence every checkstep number of steps. If None, won't
            check.
        timer : bool, optional
            Prints the time taken to execute.
        """
        start = time()
        with Pool(cpus) as pool:
            # If no initial guess has been passed, begin at the maximum likelihood
            # position.
            if p0 is None:
                tomin = lambda p: -self.model.log_likelihood(theta=p, d=self.d, 
                                                            derr=self.derr)  # TODO: should return a single value 
                res   = minimize(tomin, x0=self.model.theta_true)  # Use true params as x0.
                p0    = res.x
            elif len(np.shape(p0)) == 1:
                initpos = p0*(1 + spread*np.random.randn(nwalkers, self.model.dims))
            elif len(np.shape(p0)) == 2:
                initpos = p0
            else:
                raise ValueError('invalid p0 passed.')
            
            sampler = emcee.EnsembleSampler(
                nwalkers,
                self.model.dims, 
                self.model.log_posterior,
                args=(self.d, self.derr),
                pool=pool
            )

            # Blindly run mcmc.
            if checkstep is None:
                sampler.run_mcmc(initpos, steps, progress=progress)
                if timer:
                    print("Inference took {}s.".format(time()-start))
                if posterior_evals:
                    return sampler, self.model.post_list
                return sampler
            
            # Run mcmc, checking convergence.
            index    = 0
            autocorr = np.empty(steps)
            old_tau  = np.inf
            for sample in sampler.sample(initpos, iterations=steps, progress=progress):
                if sampler.iteration % checkstep:
                    continue
                
                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged =  np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < tol)
                if converged:
                    break
                old_tau = tau

        if timer:
            print("Inference took {}s.".format(time()-start))
        return


class SamplingNoPool:
    '''
    A base class for inference, which only uses one CPU. Great for debugging.
    Still has a cpu's argument, but ignores it.
    '''
    def __init__(self, model, d, derr):
        '''
        Parameters:
        ----------
        model: StatModel subclass
            The statistical model to use.
        d, derr: arrays
            Noisy data to use for inference.
        '''
        if not isinstance(model, StatModel):
            raise TypeError('model should be a subclass of StatModel.')
        
        if np.shape(d) != np.shape(derr):
            raise ValueError('lengths of data and its error should match.')
        
        self.model = model
        self.d     = d
        self.derr  = derr

    def inference(self, nwalkers=32, steps=5000, p0=None, spread=1e-4, 
            checkstep=None, tol=0.005, timer=True, progress=True, cpus=1):
        """
        Use Emcee to infer the model parameters given noisy data, returning an
        emcee.EnsembleSampler object that has been run.

        Parameters:
        ----------
        x, ynoisy, yerr: tuple of arrays
            Noisy y data for parameter x and its corresponding array of errors.
        nwalkers, steps: int, int
            Number of walkers and number of steps taken for Emcee.
        p0: array-like, optional
            Initial guess in parameter space. If none passed, uses the parameter
            values that maximise the likelihood.
        spread: float
            Walkers initialized in Gaussian ball with standard deviation spread.
        checkstep: int, optional
            Check convergence every checkstep number of steps. If None, won't
            check.
        timer: bool, optional
            Prints the time taken to execute.
        """
        start = time()
        # If no initial guess has been passed, begin at the maximum likelihood
        # position.
        if p0 is None:
            tomin = lambda p: -self.model.log_likelihood(theta=p, d=self.d, 
                                                        derr=self.derr)  # TODO: should return a single value 
            res   = minimize(tomin, x0=self.model.theta_true)  # Use true params as x0.
            p0    = res.x
        
        initpos = p0*(1 + spread*np.random.randn(nwalkers, self.model.dims))

        sampler = emcee.EnsembleSampler(
            nwalkers,
            self.model.dims, 
            self.model.log_posterior,
            args=(self.d, self.derr)
        )

        # Blindly run mcmc.
        if checkstep is None:
            sampler.run_mcmc(initpos, steps, progress=progress)
            if timer:
                print("Inference took {}s.".format(time()-start))
            return sampler
        
        # Run mcmc, checking convergence.
        index    = 0
        autocorr = np.empty(steps)
        old_tau  = np.inf
        for sample in sampler.sample(initpos, iterations=steps, progress=progress):
            if sampler.iteration % checkstep:
                continue
            
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged =  np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < tol)
            if converged:
                break
            old_tau = tau

        if timer:
            print("Inference took {}s.".format(time()-start))
        return sampler
