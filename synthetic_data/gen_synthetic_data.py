from numpy.random import uniform, normal
from numpy import sqrt, log, ndarray, dot, linspace, exp, vstack, ones
from scipy.special import erfinv
import matplotlib.pyplot as plt
from .cox_inverse_functions import inverse_constant_h, inverse_power_h, inverse_time_scaled_h
import numpy as np
# TODO: Do I want to make a data generator class? -> probably not because of the different parameterizations

def gen_gaussian_data(mu: float = 0, sigma: float = 1, n: int = 1000, m:int = 1) -> ndarray:
    """
    Generate gaussian data with mean mu and standard deviation sigma: p(x) = (2\pi)^{-1/2} \exp(-x^2/2)
    
    args:
        mu: mean of the gaussian
        sigma: standard deviation of the gaussian
        n: number of samples to generate
        m: number of dimensions of the gaussian
        
    returns:
        data: np.ndarray of shape (n, ) containing the generated data
    """
    # 
    # now normally this can be done by just directly sampling from a normal distribution
    # however we will do this using the inverse transform sampling method
    # first we generate a uniform distribution
    u = uniform(0, 1, (n, m))
    # then we apply the inverse transform
    return mu + sigma * sqrt(2) * erfinv(2 * u - 1)
   
def gen_exponential_data(tau: float = 1, n: int = 1000, m: int = 1) -> ndarray:
    """
    Generate exponential data with rate lam: p(x) = \tau^{-1} \exp(- x/\tau)
    
    args:
        tau: rate of the exponential
        n: number of samples to generate
        m: number of dimensions of the exponential
        
    returns:
        data: np.ndarray of shape (n, ) containing the generated data
    """
    
    u = uniform(0, 1, (n, m))
    return -tau * log(u)

def gen_cox_event_times_data(r, lambda_inv, beta: ndarray, z: ndarray, a:int = 0, b:int = 1, n:int = 1000, *args):
    """
    Helper function for generating cox event times
    
    args:
        lambda_inv: inverse of the baseline hazard rate -> this needs to be a function
        beta: regression coefficients
        z: covariates
        n: number of samples to generate
        m: number of dimensions of the data

    returns:
        clinical outcomes: (n, 2) -> (times, risk)
    """
    u = uniform(a, b, (n, 1))
    t_r = lambda_inv(-1 * exp(-1 * dot(z, beta)) * log(u), *args)
    return t_r, r * ones((n, 1)).astype(int)
    
def gen_cox_times_const_lam(r: int, lam: float, beta, z, n=1000):
    u = np.random.uniform(size=(n,1))
    t_u =  - np.divide(log(u), lam * np.exp(np.dot(z, beta))) 
    return t_u, r * ones((n, 1)).astype(int) 
        
    
# if __name__ == "__main__":
#     # test the gaussian data generator
#     # generate gaussian data
#     # mu = 0
#     # sigma = 1
#     # n = 10000
#     # data = gen_gaussian_data(mu, sigma, n)
#     # data_baseline = normal(mu, sigma, n)
#     # bins = linspace(-3, 3, 20)
#     # # plot the data
#     # plt.hist(data, bins=bins, alpha=0.5, label="inverse")
#     # plt.hist(data_baseline, bins=bins, alpha=0.5, label="baseline")
#     # plt.show()

    
#     # testing cox output
    
#     n = 10000
#     f = inverse_time_scaled_h
#     lambda_0 = 10
#     d = 10
#     p = 10.
#     tau = 0.01
    
#     # generate data
#     beta = uniform(0, 1, d)
#     z = uniform(0, 1, d)
#     data = gen_cox_event_times_data(f, beta, z, n, 1, lambda_0, tau)

#     plt.hist(data, bins=20)
#     plt.show()
