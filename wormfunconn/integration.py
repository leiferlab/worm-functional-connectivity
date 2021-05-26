import numpy as np

gregory_omega = np.array([
    np.array([5.00000000000000000000e-01]), #k=0
    np.array([4.16666666666666666670e-01,   #k=1
              1.08333333333333333330e+00]), 
    np.array([3.75000000000000000000e-01,   #k=2
              1.16666666666666666670e+00,
              9.58333333333333333330e-01]), 
    np.array([3.48611111111111111110e-01,   #k=3
              1.24583333333333333330e+00,
              8.79166666666666666670e-01,
              1.02638888888888888890e+00]),
    np.array([3.29861111111111111110e-01,   #k=4
              1.32083333333333333330e+00,
              7.66666666666666666670e-01,
              1.10138888888888888890e+00,
              9.81250000000000000000e-01]),
    np.array([3.15591931216931216930e-01,   #k=5
              1.39217923280423280420e+00,
              6.23974867724867724870e-01,
              1.24408068783068783070e+00,
              9.09904100529100529100e-01,
              1.01426917989417989420e+00]),
    np.array([3.04224537037037037040e-01,   #k=6
              1.46038359788359788360e+00,
              4.53463955026455026460e-01,
              1.47142857142857142860e+00,
              7.39393187830687830690e-01,
              1.08247354497354497350e+00,
              9.88632605820105820110e-01]),
    np.array([2.94868000440917107580e-01,   #k=7
              1.52587935405643738980e+00,
              2.56976686507936507940e-01,
              1.79890735229276895940e+00,
              4.11914406966490299820e-01,
              1.27896081349206349210e+00,
              9.23136849647266313930e-01,
              1.00935653659611992950e+00]), #k=8
    np.array([2.86975446428571428570e-01,
              1.58901978615520282190e+00,
              3.59851741622574955910e-02,
              2.24089037698412698410e+00,
              -1.40564373897707231040e-01,
              1.72094383818342151680e+00,
              7.02145337301587301590e-01,
              1.07249696869488536160e+00,
              9.92107445987654320990e-01])
    ])
    
def integral(Y, dx=1., k=8):
    '''Computes the integral of Y sampled on equispaced abscissas, using the 
    Gregory integration formula.
    
    Parameters
    ----------
    Y: numpy array
        The samples of the function to be integrated. 
    dx: scalar, optional
        The spacing between the abscissas.
    k: int, optional
        The order of the integration formula. Max: 8.
    
    Returns
    -------
    I: scalar
        The result of the integration formula.
        
    Notes: Adapted from Schueler et al. NESSi: The Non-Equilibrium Systems 
    Simulation package, arXiv:1911.01211) https://github.com/nessi-cntr/nessi
    '''
    M = Y.shape[-1]
    # Determine whether the integration order k is appropriate for the number
    # of samples. If not, pick the maximum order allowed.
    k = int(max(min(k,(M-2)//2),0))
    
    # Get the weights for the order k.
    omega = gregory_omega[k]
    
    ### Compute the integral
    I = 0.0
    # Sum the central portion of the array, leaving out the edge samples.
    I += np.sum(Y[k+1:-(k+1)])
    # Add the edge samples multiplied by the correct weights.
    if M>1:
        I += np.sum(Y[:k+1]*omega)
        I += np.sum(Y[-(k+1):]*omega[::-1])
    # Multiply by the spacing between the abscissas.
    I *= dx
    
    return I
