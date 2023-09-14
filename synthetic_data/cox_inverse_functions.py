from numpy import ndarray, linspace, exp, log, sqrt, dot

def inverse_constant_h(x, lambda_0):
    """
    Constant hazard rate function: \lambda(t) = \lambda_0
    
    args:
        x: time
        lambda_0: constant hazard rate
        
    returns:
        \Lambda^{-1}(x) = x/lambda_0
    """

    return x / lambda_0

def inverse_power_h(x, lambda_0, p=1.):
    """
    Power hazard rate function: \lambda(t) = \lambda_0 * t^p
    
    args:
        x: time
        lambda_0: constant hazard rate
        p: power
        
    returns:
        \Lambda^{-1}(x) =  (\frac{p+1}{\lambda_0}*x)^{\frac{1}{p+1}}
    """
    return (((p + 1) / lambda_0 ) * x)**(1 / (p + 1))

def inverse_time_scaled_h(x, lambda_0, tau=0.):
    """
    Time scaled hazard rate function: \lambda(t) = \lambda_0 * (t + \tau)
    
    args:
        x: time
        lambda_0: constant hazard rate
        tau: time shift
        
    returns:
        \Lambda^{-1}(x) =  \frac{1}{\lambda_0} * (\exp(\lambda_0 * x) - 1) - \tau
    """
    return (1 / tau) * (exp(x / lambda_0) - 1) 