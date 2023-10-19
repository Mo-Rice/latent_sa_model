import numpy as np
import matplotlib.pyplot as plt
from synthetic_data.gen_synthetic_data import *
from synthetic_data.cox_inverse_functions import *
from matplotlib import rcParams
from cox_regression import *

import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, jit, grad, hessian
from scipy.optimize import minimize

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 12,
})

font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 13,
        }

# simulation parameters
lambda_0 = 1 # baseline hazard
# if you don't add the second dimension such that beta_vec is a column vector, time will be a matrix
beta_vec = np.expand_dims(np.arange(-.75, 1., step=0.25), axis=1)#np.expand_dims(np.zeros((3)), axis=1 # coefficient vector
p = beta_vec.shape[0] # number of covariates
N = 2000 # number of observations
R = 1 # number of risk
z = np.random.standard_normal(size=(N, p)) # covariate matrix
tol = 1e-6 # tolerance for the optimization algorithm

t_r, r = gen_cox_times_const_lam(1, lambda_0, beta_vec, z, N)
clinical_outcomes = np.hstack((t_r, r))
filt_z_mask = np.array([(clinical_outcomes[:, 0] >= i).astype(int) for i in clinical_outcomes[:, 0]])
beta_0 =  0.*np.ones_like(beta_vec)
print('True Parameters: ', beta_vec.T)
print('Initial Parameters: ', beta_0.T)
print('-'*50)
print('Estimating regression parameters...')
beta_star, error = cox_beta_estimator(beta_0, clinical_outcomes, z)
t_max = 11
t = np.arange(0, t_max, 0.1) 
lamba_1_t = np.zeros_like(t)

for i, t_i in enumerate(t):
    lamba_1_t[i] = Lambda_r_breslow(t_i, beta_star, clinical_outcomes, z, t_max)
    
fig, ax = plt.subplots(1, 3, figsize=(12, 5))

ax[0].hist(t_r, bins=20)
ax[0].set_xlabel(r'$t$', fontdict=font)
ax[0].set_ylabel(r'$count$', fontdict=font)
ax[0].set_title( r'$\lambda(t) = \lambda_0$', fontdict=font)

ax[1].plot(t, lamba_1_t, 'k')
ax[1].plot(t, t, 'k--', linewidth=1)
ax[1].set_ylim([0, t_max - 0.5])
ax[1].set_xlim([0, t_max - 0.5])
ax[1].set_xlabel(r'$t$', fontdict=font)
ax[1].set_title(r'$\textrm{Integrated Hazard Rate}$', fontdict=font)
ax[1].set_ylabel(r'$\hat{\Lambda}_r(t)$', fontdict=font)

ax[2].errorbar(beta_vec, beta_star, yerr=error, marker='o', linestyle='None', capsize=5, color='k')
ax[2].plot(beta_vec, beta_vec, 'k--', linewidth=1)
ax[2].set_xlabel(r'$\beta^*_r$', fontdict=font)
ax[2].set_ylabel(r'$\hat{\beta}_r$', fontdict=font)
ax[2].set_title(r'$\textrm{Cox Regression Coefficients}$')

plt.tight_layout()
plt.show()
