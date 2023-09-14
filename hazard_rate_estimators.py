from jax import numpy as jnp
import numpy as np
from synthetic_data.gen_synthetic_data import gen_gaussian_data, gen_exponential_data, gen_cox_event_times_data
from synthetic_data.cox_inverse_functions import inverse_constant_h, inverse_power_h, inverse_time_scaled_h
import matplotlib.pyplot as plt

# TODO: figure out a good naming convention
# TODO: Jax implementation of the naive hazard rate estimator
def naive_hr_numpy(r, t, data: np.ndarray):
    """ 
    implementation of naive hazard rate estimator, i.e. MAP estimator for constant rates
                        \hat{h}_r(t) = \frac{\sum_{i=1}^n \delta_{rr_i}\delta(t-t_i)}{\sum_{i=1}^{n} \theta(t_i - t)}
    args:
        r: right censored indicator
        t: event time
        data: data matrix of shape (2, n)
    """
    # naive implementation using numpy
    # TODO: when estimating these will we not run into numerical rounding errors?
    # TODO this estimator is inherently flawed and hence gives bad results, implement better one pg 39 PSA
    t_max = data[data[:, 1]==r, 0].max()
    h = 0
    
    if t >= t_max:
        return np.nan
    else:
        for d in data:
            h += (d[1] == r) * (d[0] == t) / np.where((data[:, 0] <= d[0]) & (data[:, 0] < t_max))[0].shape[0]
        return h
            
# for testing purposes only
if __name__ ==  "__main__":

    beta = np.random.standard_normal()
    z = np.random.standard_normal()
    r = 2
    
    # data parameters
    lambda_0 = 1
    p = 4
    n = 2000
    
    # generate data
    # (t, r) 
    for i in range(r):
        if i == 0:
            data = gen_cox_event_times_data(i, inverse_power_h, beta, z, n, lambda_0, p) 
        else:
            data = np.vstack((data, gen_cox_event_times_data(i, inverse_power_h, beta, z, n, lambda_0, p)))

    # some scaling
    data[:, 0] = (10.*data[:, 0]).round()

    # estimate hazard rates  
    t = np.arange(0, data[:, 0].max(), step=1.)
    estimates = np.zeros(t.shape[0])
    
    for i in range(t.shape[0]):
        estimates[i] = naive_hr_numpy(1, t[i], data)
    
    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5)) 
    
    for i in range(r): 
        ax[0].hist(data[np.where(data[:,1]==i)[0], 0], bins=20, label=f"$r={i}$", alpha=0.5, density=True)
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$p(t,r)$')
    ax[0].legend()

    ax[1].plot(t, estimates)
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$\hat{h}_1(t)$')
    plt.show()
    