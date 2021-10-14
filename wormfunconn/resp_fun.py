import numpy as np

def exp_conv_2(t,p):
    '''Convolution from 0 to t of e^-t/tau1 and e^-t/tau2.'''
    A, tau1, tau2 = p
    
    y = A * tau1*tau2/(tau1-tau2) * (np.exp(-t/tau1) - np.exp(-t/tau2))

    return y 
    
def exp_conv_2b(t,p):
    '''Same as _exp_conv_2, but with alpha = gamma1 and 
    beta = gamma2-gamma1, where gamma1=1/tau1 and gamma2=1/tau2'''
    A, alpha, beta = p
    
    y = A/beta*np.exp(-alpha*t)*(1.0-np.exp(-beta*t))
    return y
