#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <iostream>
#include <complex>
#include "integration.hpp"

static PyObject *integral(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _integrationMethods[] = {
    {"integral", integral, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _integration = {
    PyModuleDef_HEAD_INIT,
    "_integration",
    NULL, // Module documentation
    -1,
    _integrationMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__integration(void) { 
        import_array(); //Numpy
        return PyModule_Create(&_integration);
    }
    
    
//////// The actual functions of the modules

static PyObject *integral(PyObject *self, PyObject *args) {

    int32_t M,k;
    double delta;
    PyObject *A_o;
    
    if(!PyArg_ParseTuple(args, "Odi", 
                &A_o, &delta, &k)) 
                return NULL;
    
    //PyArrayObject *A_a = (PyArrayObject*) PyArray_FROM_O(A_o);
    PyArrayObject *A_a = (PyArrayObject*) PyArray_FromAny(A_o,NULL,0,0,NPY_ARRAY_CARRAY,NULL);
    PyArray_Descr* info = PyArray_DESCR(A_a);
    
    // Get the PyArrayObjects. This will also cast the datatypes if needed.
    //PyArrayObject *A_a = (PyArrayObject*) PyArray_FROM_OT(A_o, NPY_FLOAT64);

    // Extract the lenghts of h and R, as their shape[0].
    M = *(PyArray_SHAPE(A_a));
    
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (A_a == NULL) {
        Py_XDECREF(A_a);
        return NULL;
    }
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    // Define the Python object that will be returned as result.
    // (Even if the result is a simple scalar, it has to be returned in a 
    // Pythonic format.)
    PyObject *result;
    
    // Depending on the type of the input, use the template function
    // integration::integral accordingly and build the actual result PyObject.
    if(PyDataType_ISFLOAT(info)){
        double *A = (double*)PyArray_DATA(A_a);
        
        double res = integration::integral<double>(A,M,delta,k);
        
        result = Py_BuildValue("d",res);
        
    } else if(PyDataType_ISCOMPLEX(info)) {
        std::complex<double> *A = (std::complex<double>*)PyArray_DATA(A_a);
        
        std::complex<double> res = integration::integral<std::complex<double> >(A,M,delta,k);
        
        result = PyComplex_FromDoubles(res.real(),res.imag());
    } else {
        result = Py_BuildValue("d",0.0);
    }
          
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(A_a);
    
    // Return the computed integral
    return result;
}
