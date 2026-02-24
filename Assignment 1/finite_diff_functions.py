import numpy as np
from scipy import sparse
from scipy.special import factorial
import matplotlib.pyplot as plt

def l2_norm(u, uhat) -> float:
    return np.sqrt(np.sum((u - uhat)**2))

def inf_norm(u, uhat) -> float:
    return np.max(np.abs(u - uhat))


def fdcoeffV_uniform(k: int, h: float, stencil: np.ndarray):
    n = len(stencil)
    A = np.ones((n,n))
    for i in range(1,n):
        A[i,:]= stencil**(i)/factorial(i)

    b = np.zeros((n,))
    b[k] = 1
    c = np.linalg.solve(A,b)
    return 1/h**k * c

def der_approx_uniform(k: int, xbar: float, h, u_func, stencil):
    coeff = fdcoeffV_uniform(k,h,stencil)
    U = u_func(xbar+h*stencil)
    #print(U)
    der = np.dot(coeff,U)
    return der

def fdcoeffV(k: int, xbar, x):
    '''
        # Parameters
        k: derivative order
        xbar: x[i]-value at which to approximate 
            k-th derivative stencil coefficients
        x: slice of x-points (x[i + stencil]) around xbar 
            to use in approximation

        # Output
        c: Coefficients of k-th order derivative of u at xbar
    '''
    n = len(x)
    A = np.ones((n,n))
    xrow = x - xbar
    for i in range(1,n):
        A[i,:]= xrow**(i)/factorial(i)

    b = np.zeros((n,))
    b[k] = 1
    c = np.linalg.solve(A,b)
    return c

def der_approx(k: int, xbar: float, x, u_func):
    '''
        # Parameters
        k: derivative order
        xbar: x[i]-value at which to approximate 
            k-th derivative stencil coefficients
        x: slice of x-points (x[i + stencil]) around xbar 
            to use in approximation
        u_func: function to approximate

        # Output
        c: Coefficients of k-th order derivative of u at xbar
    '''
    coeff = fdcoeffV(k,xbar,x)
    U = u_func(x)
    #print(U)
    der = np.dot(coeff,U)
    return der

