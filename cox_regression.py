import numpy as np
from scipy.optimize import minimize
import jax.numpy as jnp
import jax.scipy as jsp
from jax import  grad, hessian, jit
from scipy.optimize import minimize

def censor_data(c_r, data):
    """
    Function to censor the data
    
    args:
        c_r: right censoring indicator
        data: data matrix of shape (3, n)
        
    returns:
        data: censored data matrix of shape (3, n)
    """
    censor_mask = np.where(data[:, 0] > c_r)
    data[:, 1][censor_mask] = 0
    return data

def Lambda_r_breslow(t, betas_r, clinical_outcomes, covariates, t_max=None):
    """
    Brewslow estimator for the cause specific hazard rate
    
    args:
        t: time
        betas_r: regression coefficients
        clinical_outcomes: clinical outcomes matrix of shape (n, 2)
        covariates: covariates matrix of shape (n, d)
        t_max: maximum time to consider
        
    returns:
        Lam: cause specific hazard rate at time t
    """
    Lam = 0.
    times = clinical_outcomes[:, 0]
    w = np.expand_dims(np.exp(np.dot(covariates, betas_r)), axis=1)
    
    if t_max is None:
        t_max = np.max(times)
    
    if t > t_max:
        return np.infty

    failures = np.where(np.logical_and(times <= t, times <= t_max))[0]

    for t_i in failures:
        Lam +=  1 / np.sum((times >= times[t_i]).astype(int) * w.squeeze())
    return Lam

def cox_partial_ll(beta_vec, filt_t, z):
    """
    cox partial log likelihood function
    
    args:
        beta_vec: regression coefficients
        filt_t: filtered time
        z: covariates
        
    returns:
        cox partial log likelihood
    """
    beta_z = jnp.dot(z, beta_vec)
    beta_z_filt = beta_z * filt_t
    return (jsp.special.logsumexp(beta_z_filt, b=filt_t, axis=1) - beta_z).sum()/len(z)
    
def cox_beta_estimator(beta_0, clinical_outcomes, z):
    """
    Cox beta estimator
    
    args:
        beta_0: initial guess for beta
        clinical_outcomes: clinical outcomes matrix of shape (n, 2)
        z: covariates matrix of shape (n, d)
        
    returns:
        beta: beta estimator
    """
    filt_z_mask = np.array([(clinical_outcomes[:, 0] >= i).astype(int) for i in clinical_outcomes[:, 0]])
    jac = jit(grad(cox_partial_ll, argnums=0))
    hess = jit(hessian(cox_partial_ll, argnums=0))
    res = minimize(cox_partial_ll, beta_0, args=(filt_z_mask, z), jac=jac, method='BFGS')
    hess_inv = np.linalg.inv(hess(res.x, filt_z_mask, z).squeeze())
    print(res)
    return res.x, np.sqrt(np.diagonal(hess_inv)/len(z))
    