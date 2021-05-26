#include <cmath>
#include <iostream>
#include <complex>
#include "integration_coeff.tcc"

namespace integration {

template<typename T=double> T integral(T *Y, int M, double dx, int k){
    /**Computes the integral of Y sampled on equispaced abscissas, using the 
    Gregory integration formula.
    
    Parameters
    ----------
    Y: array of doubles
        The samples of the function to be integrated. 
    M: int
        Size of Y.
    dx: double, optional
        The spacing between the abscissas. Default: 1.0.
    k: int, optional
        The order of the integration formula. Max: 8. Default: 8.
    
    Returns
    -------
    I: double
        The result of the integration formula.
        
    Notes: Adapted from Schueler et al. NESSi: The Non-Equilibrium Systems 
    Simulation package, arXiv:1911.01211) https://github.com/nessi-cntr/nessi
    **/
    
    // Determine whether the integration order k is appropriate for the number
    // of samples. If not, pick the maximum order allowed.
    k = (int) std::max(std::min(k,(M-2)/2),0);
    
    // Get the pointer to the array of weights for the order k.
    double *omega = integration::gregory_omega[k];
    
    // Compute the integral
    T I = 0.0;
    // Sum the central portion of the array, leaving out the edge samples.
    for(int i=k+1;i<M-k-1;i++){I += (T) Y[i];}
    // Add the edge samples multiplied by the correct weights.
    if(M>1){
        for(int i=0;i<=k;i++){I += (T) (Y[i]+Y[M-i-1])*omega[i];}
    }
    // Multiply by the spacing between the abscissas.
    I *= (T) dx;
    
    return I;
    
}
}
