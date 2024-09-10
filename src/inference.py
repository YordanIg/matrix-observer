"""
Define likelihoods, priors, etc.
"""
import numpy as np
#from pypolychord.priors import UniformPrior

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

def prior_checker(priors, p0):
    """
    Check that each value of the vector p0 lies within the prior bounds of
    priors. If not, replace that value of the vector p0 with the mean of the
    prior bound passed.
    NOTE: You must not pass any prior bounds that average to zero.
    """
    lower =  p0 > priors[:,0]
    upper = p0 < priors[:,1]
    in_prior_pars = lower & upper
    for i in range(len(p0)):
        if not in_prior_pars[i]:
            print(f"ERROR: {i}th parameter of p0 not in prior volume. Setting to mean of prior.")
            p0[i] = np.mean(priors[i])
    return p0


################################################################################
# Polychord
################################################################################

def get_polychord_loglikelihood(y, yerr, model):
    def likelihood(theta):
        return log_likelihood(theta, y=y, yerr=yerr, model=model), []
    return likelihood

'''def get_polychord_prior(prior_pars):
    """
    The polychord prior, simply takes values in the unit hypercube and maps them
    to points in the prior range of our model.

    This prior handles an unnormalised uniform prior
    within the rectangular bounds given by prior_pars.

    inputs:
    theta - N array of parameter values
    prior_pars - [N,2] array of prior ranges
    i.e. = [[lower1, upper1], ...]
    """
    def prior(hypercube):
        return UniformPrior(prior_pars[:,0], prior_pars[:,1])(hypercube)
    return prior'''

#| Optional dumper function giving run-time read access to
#| the live points, dead points, weights and evidences

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])
