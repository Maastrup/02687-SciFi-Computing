import numpy as np
from scipy import sparse
from scipy.special import factorial
import matplotlib.pyplot as plt

def l2_norm(v) -> float:
    return np.sqrt(np.sum((v)**2))

def l2_norm_grid(v, h: float) -> float:
    return h * l2_norm(v)

def inf_norm(v) -> float:
    return np.max(np.abs(v))

def inf_norm_grid(v,h:float) -> float:
    return h**2 * inf_norm(v)


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


def smooth(U, omega, m, F):
    h = 1.0/(m+1)
    Unew = np.copy(U)

    for j in range(m):
        for i in range(m):
            k = i + j*m

            left  = U[k-1] if i > 0   else 0.0
            right = U[k+1] if i < m-1 else 0.0
            down  = U[k-m] if j > 0   else 0.0
            up    = U[k+m] if j < m-1 else 0.0

            Unew[k] = (
                (1-omega)*U[k]
                + omega/4 * (left + right + up + down - h**2 * F[k])
            )

    return Unew


def coarsen(R, m):
    mc = m // 2
    Rc = np.empty(mc**2)

    # Half weighting
    for kc in range(mc**2):
        '''
            ic = kc % mc
            jc = kc // mc
            k = 2ic + 2*jc*m 
            k = 2*kc
            k = i + j*m = 2*kc
            (i ± 1, j) = k ± 1 = 2*kc ± 1
            (i, j ± 1) = k ± m = 2*kc ± m
        '''

        ic, jc = kc % mc, kc // mc
        k = 2 * ic + 2 * jc * m

        center = R[k]
        left  = R[k-1]  if 2*ic > 0     else 0.0 # trivially true
        right = R[k+1]   if 2*ic < m-1   else 0.0
        up  = R[k-m]  if 2*jc > 0     else 0.0 # trivially true
        down    = R[k+m]   if 2*jc < m-1   else 0.0
        
        Rc[kc] = 1 / 8 * (
            left # (i - 1, j)
            + up # (i, j - 1)
            + 4 * center # (i, j)
            + down # (i, j + 1)
            + right # (i + 1, j)
        )
    return Rc


def interpolate_first_try(Rc, m):
    coarse = Rc.reshape(m//2, m//2)
    fine = np.zeros(m**2)
    fine.reshape(m,m)[::2, ::2] = coarse
    for k in range(m**2):
        i = k % m
        j = k // m
        if (i % 2 == 0) and (j % 2 == 0):
            continue
        elif (i % 2 == 0):
            up = coarse[(j - 1) // 2, i // 2]   if (j - 1) // 2 >= 0 else 0.0
            down = coarse[(j + 1) // 2, i // 2] if (j + 1) // 2 < mc else 0.0
            fine[k] = 0.5 * (up + down)
        elif (j % 2 == 0):
            left = coarse[j // 2, (i - 1) // 2]  if (i - 1) // 2 >= 0 else 0.0
            right = coarse[j // 2, (i + 1) // 2] if (i + 1) // 2 < mc else 0.0
            fine[k] = 0.5 * (left + right)
        else:
            nw = coarse[(j - 1) // 2, (i - 1) // 2]   if ((j - 1) // 2 >= 0) and ((i - 1) // 2 >= 0) else 0.0
            sw = coarse[(j + 1) // 2, (i - 1) // 2] if ((j + 1) // 2 < mc) and ((i - 1) // 2 >= 0) else 0.0
            ne = coarse[(j - 1) // 2, (i + 1) // 2] if ((j - 1) // 2 >= 0) and ((i + 1) // 2 < mc) else 0.0
            se = coarse[(j + 1) // 2, (i + 1) // 2] if ((j + 1) // 2 < mc) and ((i + 1) // 2 < mc) else 0.0
            fine[k] = 0.25 * (nw + sw + ne + se)
    return fine


def interpolate_first_try(Rc, m):
    coarse = Rc.reshape(m//2, m//2)
    fine = np.zeros(m**2)
    fine.reshape(m,m)[::2, ::2] = coarse
    for k in range(m**2):
        i = k % m
        j = k // m
        if (i % 2 == 0) and (j % 2 == 0):
            continue
        elif (i % 2 == 0):
            up = coarse[(j - 1) // 2, i // 2]   if (j - 1) // 2 >= 0 else 0.0
            down = coarse[(j + 1) // 2, i // 2] if (j + 1) // 2 < mc else 0.0
            fine[k] = 0.5 * (up + down)
        elif (j % 2 == 0):
            left = coarse[j // 2, (i - 1) // 2]  if (i - 1) // 2 >= 0 else 0.0
            right = coarse[j // 2, (i + 1) // 2] if (i + 1) // 2 < mc else 0.0
            fine[k] = 0.5 * (left + right)
        else:
            nw = coarse[(j - 1) // 2, (i - 1) // 2]   if ((j - 1) // 2 >= 0) and ((i - 1) // 2 >= 0) else 0.0
            sw = coarse[(j + 1) // 2, (i - 1) // 2] if ((j + 1) // 2 < mc) and ((i - 1) // 2 >= 0) else 0.0
            ne = coarse[(j - 1) // 2, (i + 1) // 2] if ((j - 1) // 2 >= 0) and ((i + 1) // 2 < mc) else 0.0
            se = coarse[(j + 1) // 2, (i + 1) // 2] if ((j + 1) // 2 < mc) and ((i + 1) // 2 < mc) else 0.0
            fine[k] = 0.25 * (nw + sw + ne + se)
    return fine