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
    t_max = data[data[:, 1]==r, 0].max()
    h = 0
    
    if t >= t_max:
        return np.infty
    else:
        for d in data:
            h += (d[1] == r) * (d[0] == t) / np.where((data[:, 0] <= d[0]) & (data[:, 0] < t_max))[0].shape[0]
        return h

def naive_S_numpy(hr_estimates, r=None):
    """
    implementation of naive survival function estimator
                        \hat{S}(t) = e^{-sum_i \frac{\theta(t - t_i)}{\sum_j \theta (t_j - t)}}
    args:
        estimates: hazard rate estimates of shape (r, t)
    """
    if r is None:
        return np.exp(-hr_estimates.cumsum(axis=1).sum(axis=0))
    else: 
      return np.exp(-hr_estimates[r, :].cumsum(axis=1).sum(axis=0))

def S_km_numpy(data):
    """
    implementation of Kaplan-Meier estimator
                        \hat{S}(t) = \prod_{t_i < t} (1 - \frac{d_i}{n_i})
    args:
        data: data matrix of shape (2, n)
    """
    n = data.shape[0]
    t = data[:, 0]
    r = data[:, 1]
    S = np.ones(t.shape[0])
    
    for i in range(t.shape[0]):
        S[i] = np.prod(1 - r[t <= t[i]] / n)
    
    return S
 
# for testing purposes only
if __name__ ==  "__main__":

    beta = np.random.standard_normal()
    z = np.random.standard_normal()
    r = 2
    
    # data parameters
    lambda_0 = 1
    p = 2
    n = 500
    data_function = inverse_power_h
    tau = 1
    
    # generate data
    # (t, r) 
    for i in range(r):
        if i == 0:
            data = gen_cox_event_times_data(i, data_function, beta, z, n, lambda_0, p) 
        else:
            data = np.vstack((data, gen_cox_event_times_data(i, data_function, beta, z, n, lambda_0, p)))


    # some scaling
    data[:, 0] = (10.*data[:, 0]).round()

    # estimate hazard rates  
    t = np.arange(0, 50, step=1.)
    estimates = np.zeros((r,t.shape[0])) # doing this for all r's to calculate survival function
    
    for i in range(t.shape[0]):
        estimates[0][i] = naive_hr_numpy(0, t[i], data)
        estimates[1][i] = naive_hr_numpy(1, t[i], data)
    
    # estimate survival function
    S_t = naive_S_numpy(estimates)
    
    # plotting
    fig, ax = plt.subplots(1, 3, figsize=(13, 5)) 
    
    for i in range(r): 
        ax[0].hist(data[np.where(data[:,1]==i)[0], 0], bins=20, label=f"$r={i}$", alpha=0.5, density=True)
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$p(t,r)$')
    ax[0].legend()

    ax[1].plot(t, estimates[0, :], label=r'$\hat{h}_0(t)$')
    ax[1].plot(t, estimates[1, :], label=r'$\hat{h}_1(t)$')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$\hat{h}_r(t)$')
    ax[1].legend()

    ax[2].plot(t, S_t)
    ax[2].set_xlabel(r'$t$')
    ax[2].set_ylabel(r'$S(t)$')
    
    plt.tight_layout()
    plt.show()
    