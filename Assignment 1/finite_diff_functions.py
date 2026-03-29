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

def vec_smooth(U, omega, m, F):
    h = 1 / (m+1)
    Ugrid = np.pad(U.reshape((m,m)), ((1,1), (1,1)), 'constant')
    Fgrid = np.pad(F.reshape((m,m)), ((1,1), (1,1)), 'constant')
    Ugrid[1:-1, 1:-1] = (1 - omega) * Ugrid[1:-1, 1:-1] + omega * 0.25 * (
        Ugrid[:-2, 1:-1] + Ugrid[1:-1, :-2]
        + Ugrid[2:, 1:-1] + Ugrid[1:-1, 2:]
        - h**2 * Fgrid[1:-1, 1:-1]
    )
    return Ugrid[1:-1, 1:-1].reshape(-1)

def coarsen(R, m):
    mc = m // 2
    Rc = np.empty(mc**2)

    # Half weighting
    for i in range(mc):
        for j in range(mc):
            '''
                ic = kc % mc
                jc = kc // mc
                k = 2ic + 2*jc*m 
                k = 2*kc
                k = i + j*m = 2*kc
                (i ± 1, j) = k ± 1 = 2*kc ± 1
                (i, j ± 1) = k ± m = 2*kc ± m
            '''

            kc = i + j*mc
            k = (2*i + 1) + (2*j + 1)*m

            center = R[k]
            left  = R[k-1] 
            right = R[k+1] 
            up  = R[k-m]   
            down    = R[k+m]
            
            Rc[kc] = 1 / 8 * (
                left # (i - 1, j)
                + up # (i, j - 1)
                + 4 * center # (i, j)
                + down # (i, j + 1)
                + right # (i + 1, j)
            )
    return Rc

def index_c_to_f(i,j):
    return (2*i + 1, 2*j + 1)

def scatter(Rc, m):
    mc = m // 2
    fine = np.zeros(m**2)
    fine.reshape(m,m)[1::2, 1::2] = Rc.reshape(mc, mc)
    for ic in range(mc):
        for jc in range(mc):
            i, j = index_c_to_f(ic, jc)
            # fine indexes
            n = i + (j-1) * m
            s = i + (j + 1) * m
            w = i - 1 + j*m
            e = i + 1 + j*m
            nw = i - 1 + (j-1) * m
            ne = i + 1 + (j-1) * m
            sw = i - 1 + (j+1) * m
            se = i + 1 + (j+1) * m
            corners = np.array([nw, ne, sw, se])
            cross = np.array([n, s, w, e])

            center_val = Rc[ic + jc*mc]
            fine[corners] += 1/4 * center_val
            fine[cross] += 1/2 * center_val
    return fine