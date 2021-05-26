#include <cmath>
#include <iostream>
#include <stdint.h>
#include "convolution.hpp"
#include "integration.hpp"

double _convolution1(double *A, double *B, int t, double delta, int k) {
    double y = 0.0;
    double *C = new double[t+1];

    for(int s=0;s<=t;s++){C[s] = A[t-s]*B[s];}    
    y = integration::integral<double>(C,t+1,delta,k);
    
    delete[] C;
    return y;
}

// Overload to allow to choose convolution or convolution_add via the argument bool add
void convolution(double *A, double *B, int M, double delta, double *out, bool add, int k) {
    for(int m=0;m<M;m++){
        if(add){out[m] += _convolution1(A,B,m,delta,k);}
        else {out[m] = _convolution1(A,B,m,delta,k);}
    }
}

void convolution(double *A, double *B, int M, double delta, double *out, int k) {
    for(int m=0;m<M;m++){
        out[m] = _convolution1(A,B,m,delta,k);
    }
}


void convolution_add(double *A, double *B, int M, double delta, double *out, int k) {
    for(int m=0;m<M;m++){
        out[m] += _convolution1(A,B,m,delta,k);
    }
}
