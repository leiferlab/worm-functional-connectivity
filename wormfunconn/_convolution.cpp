#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <iostream>
#include "convolution.hpp"

static PyObject *convolution(PyObject *self, PyObject *args);
static PyObject *convolution1(PyObject *self, PyObject *args);
static PyObject *slice_test(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _convolutionMethods[] = {
    {"convolution", convolution, METH_VARARGS, ""},
    {"convolution1", convolution1, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _convolution = {
    PyModuleDef_HEAD_INIT,
    "_convolution",
    NULL, // Module documentation
    -1,
    _convolutionMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__convolution(void) { 
        import_array(); //Numpy
        return PyModule_Create(&_convolution);
    }
    
    
//////// The actual functions of the modules

static PyObject *convolution1(PyObject *self, PyObject *args) {

    int32_t M,N,k;
    double delta;
    PyObject *A_o, *B_o;
    
    if(!PyArg_ParseTuple(args, "OOidi", 
                &A_o, &B_o, &N, &delta, &k)) 
                return NULL;
    
    // Get the PyArrayObjects. This will also cast the datatypes if needed.
    PyArrayObject *A_a = (PyArrayObject*) PyArray_FROM_OT(A_o, NPY_FLOAT64);
    PyArrayObject *B_a = (PyArrayObject*) PyArray_FROM_OT(B_o, NPY_FLOAT64);
    
    // Extract the lenghts of A, as its shape[0].
    M = *(PyArray_SHAPE(A_a));
    if(N>(M-1)){
        Py_XDECREF(A_a);
        Py_XDECREF(B_a);
        Py_XINCREF(Py_None); 
        return Py_None;
    }
    
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (A_a == NULL || B_a == NULL) {
        Py_XDECREF(A_a);
        Py_XDECREF(B_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    double *A = (double*)PyArray_DATA(A_a);
    double *B = (double*)PyArray_DATA(B_a);
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    double result = 0.0;
    result = _convolution1(A,B,N,delta,k);
          
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(A_a);
    Py_XDECREF(B_a);
    
    PyObject *result_o = Py_BuildValue("d",result);
    
    // Return the computed Fourier integral
    return result_o;
}

static PyObject *convolution(PyObject *self, PyObject *args) {

    int32_t M,k;
    double delta;
    PyObject *A_o, *B_o, *out_o;
    
    if(!PyArg_ParseTuple(args, "OOdi", 
                &A_o, &B_o, &delta, &k)) 
                return NULL;
    
    // Get the PyArrayObjects. This will also cast the datatypes if needed.
    PyArrayObject *A_a = (PyArrayObject*) PyArray_FROM_OT(A_o, NPY_FLOAT64);
    PyArrayObject *B_a = (PyArrayObject*) PyArray_FROM_OT(B_o, NPY_FLOAT64);
    
    // Extract the lenghts of h and R, as their shape[0].
    M = *(PyArray_SHAPE(A_a));
    
    // Create the numpy array to be returned
    out_o = PyArray_SimpleNew(1, PyArray_SHAPE(A_a), NPY_FLOAT64);
    PyArrayObject *out_a = (PyArrayObject*) PyArray_FROM_OT(out_o, NPY_FLOAT64);
    Py_INCREF(out_o);
    
        
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (A_a == NULL || B_a == NULL || out_a == NULL) {
        Py_XDECREF(A_a);
        Py_XDECREF(B_a);
        Py_XDECREF(out_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    double *A = (double*)PyArray_DATA(A_a);
    double *B = (double*)PyArray_DATA(B_a);
    double *out = (double*)PyArray_DATA(out_a);
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////

    convolution(A,B,M,delta,out,k);
          
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(A_a);
    Py_XDECREF(B_a);
    Py_XDECREF(out_a);
    
    // Return the computed Fourier integral
    return out_o;
}
