"""
Define likelihoods, priors, etc.
"""
import numpy as np

################################################################################
# Emcee
################################################################################

def log_likelihood(theta, y, yerr, model):
    """
    Compute the Gaussian log-likelihood, given a model(theta) and data y
    with errors yerr.
    """
    y_model = model(theta)
    chi2 = (y - y_model)**2 / (yerr**2)
    return np.sum(-chi2 / 2)

def log_prior(theta, prior_pars):
    """
    Log Prior probability

        log(prior(theta))

        This handles an unnormalised uniform prior
        within the rectangular bounds given by prior_pars.

        inputs:
        theta - N array of parameter values
        prior_pars - [N,2] array of prior ranges
        i.e. = [[lower1, upper1], ...]

        Returns 0 if theta in prior_pars and -inf otherwise
    """

    lower =  theta > prior_pars[:,0]
    upper = theta < prior_pars[:,1]
    in_prior_pars = all(lower & upper)

    #return prior value
    if in_prior_pars:
        return 0.0
    return -np.inf

def log_prior_gauss(theta, prior_pars):
    """
    Log Prior probability

        log(prior(theta))

        This handles an unnormalised gaussian prior
        with the parameters prior_pars = ndarray([(mean, std), (mean, std), ...])
        for each of the parameters of theta.
    """
    means = prior_pars[:,0]
    stds  = prior_pars[:,1]
    return np.sum(-.5 * ((theta-means)/stds)**2)

def log_posterior(theta, y, yerr, model, prior_pars):
    lp = log_prior(theta, prior_pars)
    if np.isfinite(lp):
        lp += log_likelihood(theta, y, yerr, model)
    return lp

################################################################################
# Polychord
################################################################################
